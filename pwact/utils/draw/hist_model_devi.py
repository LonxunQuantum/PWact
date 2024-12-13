import os
import numpy as np

# import numpy as np
import matplotlib.pyplot as plt

mark_list = ["o", "^", "v", "X", "|", "*", "v", "+", '*', ' ']
color_list = ["#006400", "#FF8C00", "#B22222", "#FF8C00" ,"#ff0099" ,\
              "#000000" , "#BDB76B", "#999900" ,"#009999" ,"#000099" ,"#990099", "#ff9900" ]
linestyle_list = ["--", "--", "--", "-", "-", "-"]

def draw_hists_2(data, save_path, min, max, legend_label):
    bins = np.linspace(0, max, 100)  # 根据需要调整bin的数量和范围
    alpha = 0.5  # 透明度设置
    plt.figure(figsize=(10, 6))  # 设置图形的大小

    for i, _data in enumerate(data):
        plt.hist(_data, bins=bins, alpha=alpha, label=legend_label[i], density=True)

    # 设置坐标轴范围
    plt.xlim(min, max)
    
    # 添加标题、标签、图例
    plt.title('Distribution of model deviation', fontsize=18, fontweight='bold', family='serif')
    plt.xlabel('Model Deviation', fontsize=14, fontweight='bold', family='serif')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold', family='serif')
    
    # 设置图例
    plt.legend(loc='upper right', fontsize=15, title_fontsize='large')

    # 调整坐标轴的字体大小
    plt.tick_params(axis='both', labelsize=15)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def read_model_devi(file_path, low=None, high=None):
    devi_files = []
    for path, dirList, fileList in os.walk(file_path):
        for _ in fileList:
            if "model_devi.out" in _:
                devi_files.append(os.path.join(path, _))
    data = []
    for file in devi_files:
        _data = np.loadtxt(file, skiprows=0)
        if _data.shape[0] > 1:
            data.extend(list(_data[:, 1])) # force
    
    right = 0
    mid = 0
    error = 0
    if low is not None:
        for d in data:
            if d <= low:
                right += 1
            elif d > low and d <= high:
                mid += 1
            else:
                error += 1
    return data, devi_files, right, mid, error
    
def draw_hist_list(file_path:list[str], legend_label, save_path:str, low=None, high=None) :
    data = []
    min = None
    max = None
    abs_mean = []
    abs_max = []
    mid = 0
    error = 0
    right = 0
    for file in file_path:
        _data, devi_files, _right, _mid, _error = read_model_devi(file, low, high)
        right += _right
        mid += _mid
        error += _error
        _min = np.min(_data)
        _max = np.max(_data)
        min = _min if (min is None or _min < min) else min
        max = _max if (max is None or _max > max) else max
        data.append(_data)
        abs_mean.append(np.mean(_data))
        abs_max.append(np.max(_data))
    
    for i in range(0, len(legend_label)):
        if low is not None:
            legend_label[i] = "{}\nmean {} max {} right {} candidate {} error {}".format(legend_label[i], round(abs_mean[i], 3), round(abs_max[i], 3), right, mid, error)
        else:
            legend_label[i] = "{}\nmean {} max {}".format(legend_label[i], round(abs_mean[i], 5), round(abs_max[i], 5))
    # print(legend_label)
    new_data = []
    for id, d in enumerate(data):
        tmp = []
        for _d in d:
            if _d > 1.0:
                tmp.append(1.0)
            else:
                tmp.append(_d)
        new_data.append(tmp)
    max = 1.0 if max > 1.0 else max
    draw_hists_2(new_data, save_path, min, max, legend_label)




if __name__ == "__main__":
    legend_label = ["iter.0006/md/md.000.sys.000"]
    save_dir = "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.002"
    draw_hist_list([
        "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.002"
    ], legend_label, save_path=os.path.join(save_dir, "hist_mode_deiv_md.000.sys.002.png"))

    legend_label = ["iter.0006/md/md.000.sys.001"]
    save_dir = "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.000"
    draw_hist_list([
        "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.000"
    ], legend_label, save_path=os.path.join(save_dir, "hist_mode_deiv_md.000.sys.000.png"))

    legend_label = ["iter.0006/md/md.000.sys.002"]
    save_dir = "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.001"
    draw_hist_list([
        "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.001"
    ], legend_label, save_path=os.path.join(save_dir, "hist_mode_deiv_md.000.sys.001.png"))

    legend_label = ["iter.0006/md/md.000.sys.003"]
    save_dir = "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.003"
    draw_hist_list([
        "/data/home/wuxingxing/datas/al_dir/djp/iter.0006/explore/md/md.000.sys.003"
    ], legend_label, save_path=os.path.join(save_dir, "hist_mode_deiv_md.000.sys.003.png"), low=0.1, high=0.2)
