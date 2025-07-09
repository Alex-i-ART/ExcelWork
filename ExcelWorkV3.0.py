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

def get_user_input():
    month = input("Введите месяц (например, 'Июль'): ").strip()
    day = input("Введите число (например, '15'): ").strip()
    return month, day

def read_excel_data(file_path, month, day):
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
        skiprows=50,
        nrows=1
    )
    line_data["Факт"] = [int(round(x)) if isinstance(x, (int, float)) else x for x in line_df.iloc[0, 1:]]
    line_data["Факт"].insert(0,0)
    print(line_data["Факт"])

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

def update_bar_chart(ax, month, day, start_idx, visible_categories=15):
    ax.clear()
    
    ax.set_position([0.30, 0.3, 0.55, 0.6])
    end_idx = min(start_idx + visible_categories, len(bar_categories))
    current_categories = bar_categories[start_idx:end_idx]
    current_values = bar_values[start_idx:end_idx]
    procent = round(float(day)/float(max(x_values))*100,2)
    
    colors = []
    print(current_values)
    for value in current_values:
        if value*100 > procent:
            colors.append("#00FF3C")
        elif value*100 == procent:
            colors.append("#3700FF")
        else:
            colors.append("#FF0008")
    
    bars = ax.barh(current_categories, current_values, color=colors, edgecolor='gray', alpha=0.8)
    
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%","25%", "50%","75%", "100%"], fontweight='bold')
    ax.set_title(f'ВЫПОЛНЕНИЕ ПЛАНА ПО ПРИЗЫВУ ЗА {month.upper()}', 
                fontsize=11, pad=20, fontweight='bold', color='red')
    ax.set_xlabel('% ВЫПОЛНЕНИЯ', fontsize=11, fontweight='bold')
    ax.set_ylabel('АДМИНИСТРАТИВНЫЕ ЦЕНТРЫ', fontsize=11, fontweight='bold')
    
    for i, v in enumerate(current_values):
        ax.text(v + 0.01, i, f"{v:.2%}", 
                va='center', color=colors[i], fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.grid(axis='x', linestyle='--', alpha=0.6)

def update_line_chart(ax, month):
    ax.clear()
    
    colors = ["#0062FF", "#FF5E00", "#00FF3C"]
    markers = ['o', 's', 'D']
    line_styles = ['-', '-', '-']
    
    for i, (name, values) in enumerate(line_data.items()):
        ax.plot(x_values, values, 
                label=name.upper(), 
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                markersize=0,
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
                            fontweight='bold',
                            color=colors[i % len(colors)],
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            prev_value = y
    
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_title(f'ВЫПОЛНЕНИЕ ПЛАНА ПО ОТБОРУ КАНДИДАТОВ НА ВОЕННУЮ СЛУЖБУ ПО КОНТРАКТУ В ВООРУЖЕННЫЕ СИЛЫ РОССИЙСКОЙ ФЕДЕРАЦИИ В ЗАБАЙКАЛЬСКОМ КРАЕ ЗА {month.upper()} 2025', 
                fontsize=12, pad=20, wrap=True, fontweight='bold', color='red')
    ax.set_xlabel(month.upper(), fontsize=10, fontweight='bold', color='red')
    ax.set_ylabel('КОЛИЧЕСТВО ГРАЖДАН, ПОДПИСАВШИХ КОНТРАКТ', fontsize=12, wrap=True, fontweight='bold')
    
    ax.legend(frameon=True, framealpha=0.9, shadow=True, borderpad=1, prop={'weight':'bold'})
    ax.grid(True, linestyle='--', alpha=0.5)
    
    for i, (name, values) in enumerate(line_data.items()):
        ax.annotate(f'{name.upper()}: {values[-1]}', 
                    xy=(x_values[-1], values[-1]),
                    xytext=(-35, 10), 
                    textcoords='offset points',
                    fontweight='bold',
                    color=colors[i % len(colors)],
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->'))

def update_slider(val):
    update_bar_chart(ax_bar, month, day, int(val))
    fig.canvas.draw_idle()

def update_graphs(frame):
    if frame % 2 == 0:
        # Показываем столбчатую диаграмму и слайдер
        ax_bar.set_visible(True)
        ax_slider.set_visible(True)
        ax_line.set_visible(False)
        update_bar_chart(ax_bar, month, day, int(slider.val))
    else:
        # Показываем линейную диаграмму и скрываем слайдер
        ax_bar.set_visible(False)
        ax_slider.set_visible(False)
        ax_line.set_visible(True)
        update_line_chart(ax_line, month)
    
    plt.subplots_adjust(left=0.3, right=0.85, top=0.9, bottom=0.2)

if __name__ == "__main__":
    try:
        month, day = get_user_input()
        excel_file = find_excel_file()
        print(f"Анализируем файл: {excel_file}")
        print(f"Используется месяц: {month}")
        
        (bar_categories, bar_values), (x_values, line_data) = read_excel_data(excel_file, month, day)
        
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
            label='КАТЕГОРИИ',
            valmin=0,
            valmax=max(0, len(bar_categories) - 15),
            valinit=0,
            valstep=1,
            orientation='vertical',
            color='red'
        )
        slider.label.set_fontweight('bold')
        slider.on_changed(update_slider)
        
        # Изначально показываем столбчатую диаграмму
        ax_line.set_visible(False)
        update_bar_chart(ax_bar, month, day, 0)
        
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