import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import seaborn as sns
from matplotlib.widgets import Slider

# Устанавливаем стиль seaborn
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-pastel')

def distribute_integer(total, days=30):
    base = total // days
    remainder = total % days
    daily_values = [base + 1 if i < remainder else base for i in range(days)]
    
    cumulative_sum = 0
    cumulative_values = []
    for value in daily_values:
        cumulative_sum += value
        cumulative_values.append(cumulative_sum)
    
    return cumulative_values

def find_excel_file():
    files = glob.glob('*.xlsx')
    if not files:
        raise FileNotFoundError("Не найден файл с расширением .xlsx")
    return files[0]

def read_month_from_txt():
    try:
        with open('month.txt', 'r', encoding='utf-8') as file:
            month = file.read().strip()
            month = month[1:]
            return month
    except FileNotFoundError:
        print("Файл month.txt не найден. Используется значение по умолчанию 'Июль'")
        return "Июль"

def read_excel_data(file_path, month):
    bar_df = pd.read_excel(
        file_path, 
        header=None,
        usecols="C,G",
        skiprows=[0,1,2,3,14],
        nrows=35
    ).sort_values(6, ascending=True)

    bar_categories = bar_df.iloc[:, 0].tolist()
    bar_values = [float(x) for x in bar_df.iloc[:, 1].tolist()]
    
    line_data = {}
    
    line_df = pd.read_excel(
        file_path,
        header=None,
        usecols="J:AN",
        skiprows=39,
        nrows=1
    )
    line_data["Факт"] = line_df.iloc[0,:].cumsum().tolist()

    line_df = pd.read_excel(
        file_path,
        header=None,
        usecols="J:AN",
        skiprows=2,
        nrows=1
    )
    x_values = line_df.dropna(axis=1).iloc[0,:].tolist()

    diff = len(line_data["Факт"]) - len(x_values)
    if diff == 1:
        line_data["Факт"] = line_data["Факт"][:-1]
    elif diff == -2:
        line_data["Факт"] = line_data["Факт"][:-2]

    line_df = pd.read_excel(
        file_path,
        header=None,
        usecols="D",
        skiprows=3,
        nrows=1
    )
    plan_value = line_df.iloc[0, 0]
    line_data["План ВВО"] = distribute_integer(plan_value, len(x_values))
    line_data["План НЦУО"] = distribute_integer(285, len(x_values))
     
    return (bar_categories, bar_values), (x_values, line_data)

def update_bar_chart(ax, month, start_idx, visible_categories=15):
    ax.clear()
    
    ax.set_position([0.30, 0.3, 0.55, 0.6])
    end_idx = min(start_idx + visible_categories, len(bar_categories))
    current_categories = bar_categories[start_idx:end_idx]
    current_values = bar_values[start_idx:end_idx]
    
    max_value = max(bar_values)
    min_value = min(bar_values)
    threshold_high = max_value * 0.7
    threshold_low = max_value * 0.3
    
    colors = []
    for value in current_values:
        if value >= threshold_high:
            colors.append('#55A868')
        elif value >= threshold_low:
            colors.append('#DD8452')
        else:
            colors.append('#C44E52')
    
    bars = ax.barh(current_categories, current_values, color=colors, edgecolor='gray', alpha=0.8)
    
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%","25%", "50%","75%", "100%"])
    ax.set_title(f'Выполнение плана по призыву за {month.lower()}', fontsize=11, pad=20)
    ax.set_xlabel('% выполнения', fontsize=11)
    ax.set_ylabel('Административные центры', fontsize=11)
    
    for i, v in enumerate(current_values):
        ax.text(v + 0.01, i, f"{v:.2%}", 
                va='center', color='black', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.grid(axis='x', linestyle='--', alpha=0.6)

def update_line_chart(ax, month):
    ax.clear()
    

    colors = ['#4C72B0', '#DD8452', '#55A868']
    markers = ['o', 's', 'D']
    line_styles = ['-', '--', '-.']
    
    for i, (name, values) in enumerate(line_data.items()):
        ax.plot(x_values, values, 
                label=name, 
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=0.5)
        
        threshold = max(values) * 0.10
        
        prev_value = None
        for x, y in zip(x_values, values):
            if prev_value is not None and abs(y - prev_value) >= threshold:
                ax.annotate(f'{y}',
                            xy=(x, y),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            prev_value = y
    
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=45, ha='right', fontsize=10)
    ax.set_title(f'Выполнение плана по отбору кандидатов на военную службу по контракту в Вооруженные Силы Российской Федерации В Забайкальском крае за {month.upper()} 2025', fontsize=12, pad=20, wrap=True)
    ax.set_xlabel(month, fontsize=10)
    ax.set_ylabel('Количество граждан, подписавших контракт', fontsize=12, wrap=True)
    
    ax.legend(frameon=True, framealpha=0.9, shadow=True, borderpad=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    for name, values in line_data.items():
        ax.annotate(f'{name}: {values[-1]}', 
                    xy=(x_values[-1], values[-1]),
                    xytext=(-35, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->'))

def update_slider(val):
    update_bar_chart(ax_bar, month, int(val))
    fig.canvas.draw_idle()

def update_graphs(frame):
    if frame % 2 == 0:
        # Показываем столбчатую диаграмму и слайдер
        ax_bar.set_visible(True)
        ax_slider.set_visible(True)
        ax_line.set_visible(False)
        update_bar_chart(ax_bar, month, int(slider.val))
    else:
        # Показываем линейную диаграмму и скрываем слайдер
        ax_bar.set_visible(False)
        ax_slider.set_visible(False)
        ax_line.set_visible(True)
        update_line_chart(ax_line, month)
    
    plt.subplots_adjust(left=0.3, right=0.85, top=0.9, bottom=0.2)

if __name__ == "__main__":
    try:
        month = read_month_from_txt()
        excel_file = find_excel_file()
        print(f"Анализируем файл: {excel_file}")
        print(f"Используется месяц: {month}")
        
        (bar_categories, bar_values), (x_values, line_data) = read_excel_data(excel_file, month)
        
        print("\nДанные для столбчатой диаграммы:")
        print(f"Категории ({len(bar_categories)}): {bar_categories[:3]}...")
        print(f"Значения: {[f'{x:.2%}' for x in bar_values[:3]]}...")
        
        print("\nДанные для линейного графика:")
        for name, values in line_data.items():
            print(f"{name}: {values[:5]}... (всего {len(values)} значений)")
        
        fig = plt.figure(figsize=(14, 8), facecolor='#f5f5f5')
        
        # Создаем оси для графиков и слайдера
        ax_bar = fig.add_axes([0.1, 0.2, 0.7, 0.7])  # Основная область для столбчатой диаграммы
        ax_slider = fig.add_axes([0.90, 0.2, 0.03, 0.7])  # Вертикальный слайдер справа
        ax_line = fig.add_axes([0.1, 0.2, 0.85, 0.7])    # Область для линейного графика
        
        # Настраиваем вертикальный слайдер
        slider = Slider(
            ax=ax_slider,
            label='Категории',
            valmin=0,
            valmax=max(0, len(bar_categories) - 15),
            valinit=0,
            valstep=1,
            orientation='vertical'
        )
        slider.on_changed(update_slider)
        
        # Изначально показываем столбчатую диаграмму
        ax_line.set_visible(False)
        update_bar_chart(ax_bar, month, 0)
        
        ani = FuncAnimation(
            fig, 
            update_graphs,
            frames=2, 
            interval=5000,
            repeat=True,
            cache_frame_data=False
        )
        
        plt.show()
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Проверьте: 1) наличие Excel-файла, 2) структуру данных в файле")