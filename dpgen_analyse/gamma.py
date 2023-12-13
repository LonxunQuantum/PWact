import argparse
import os
import matplotlib.pyplot as plt

# color_list = ["#000000", "#BDB76B", "#B8860B", "#008B8B", "#FF8C00", "#A52A2A", "#5F9EA0"]
#黑，灰色，浅棕色，浅黄色，深棕色，深黄色，深绿色
# color_list = ["#000000", "#A9A9A9", "#BDB76B", "#F0E68C", "#483D8B", "#DAA520", "#008000"]
color_list = ["#008000", "#BDB76B", "#DAA520"]
# color_list = ["#BDB76B", "#008B8B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+", "o", '*']

def draw_lines(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None, \
                        xticks:list=None, xtick_loc:list=None):
    # force-kpu散点图
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 20,
        }
    plt.figure(figsize=(12,9))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    for i in range(len(y_list)):
        # if i % 2 != 0:
        #     continue
        plt.plot(x_list[i], y_list[i], \
            color=color_list[i], marker=mark_list[i], \
                label=legend_label[i])
    
    if xticks is not None:
        plt.xticks(xtick_loc, xticks, fontsize=14)
    plt.xlabel(x_label, font)
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel(y_label, font)
    plt.title(title, font)
    plt.legend(fontsize=12, frameon=False, loc=location)
    plt.savefig(picture_save_path)

def calculate_params(save_dir,iters,lambda0, v0):
    lambda_list = []
    lambda_list.append(lambda0)
    for t in range(1, iters):
        lambda_list.append(v0 * lambda_list[t-1] + 1 - v0)

    alpha_list = []
    secp_alpha_list= []
    alpha_list.append(lambda0**-0.5)
    secp_alpha_list.append(alpha_list[0]**-2)
    for t in range(1, iters):
        alpha_list.append(alpha_list[t-1]*lambda_list[t]**-0.5)
        secp_alpha_list.append(alpha_list[t]**-2)

    gamma_list = []
    secp_alpha_gamma_list = []
    gamma_list.append(1)
    for t in range(1, iters):
        gamma_list.append((gamma_list[t-1])/(lambda_list[t] + secp_alpha_list[t] * gamma_list[t-1]))
        secp_alpha_gamma_list.append(secp_alpha_list[t]*gamma_list[t-1])

    beta_list = []
    for t in range(1, iters):
        beta_list.append((lambda_list[t])/(lambda_list[t] + alpha_list[t]**(-2) * gamma_list[t-1]))

    xrange = list(range(0, len(gamma_list), 1))

    draw_lines([xrange], [gamma_list], [r"$\gamma_t$"], \
    x_label = "iters", y_label = r"$\gamma$", \
        title = r"$\gamma_t$ with iter increase", location = "best", \
            picture_save_path = os.path.join(save_dir, "gamma_t_pic1_{}.png".format(iters)))
    
    xrange = list(range(0, len(beta_list), 1))
    draw_lines([xrange], [beta_list], [r"$\beta_t$"], \
    x_label = "iters", y_label = r"$\beta$", \
        title = r"$\beta_t$ with iter increase", location = "best", \
            picture_save_path = os.path.join(save_dir, "beta_t_pic2_{}.png".format(iters)))

    xrange = list(range(0, len(secp_alpha_list), 1))
    draw_lines([xrange], [secp_alpha_list], [r"$\alpha_t^{-2}$"], \
    x_label = "iters", y_label = r"$\alpha^{-2}$", \
        title = r"$\alpha_t^{-2}$ with iter increase", location = "best", \
            picture_save_path = os.path.join(save_dir, "alpha_secp_pic3_{}.png".format(iters)))
    
    xrange = list(range(0, len(secp_alpha_gamma_list), 1))
    draw_lines([xrange], [secp_alpha_gamma_list], [r"$\alpha_t^{-2}\cdot \gamma_{t-1}$"], \
    x_label = "iters", y_label = r"$\alpha_t^{-2}\cdot \gamma_{t-1}$", \
        title = r"$\alpha_t^{-2}\cdot \gamma_{t-1}$ with iter increase", location = "best", \
            picture_save_path = os.path.join(save_dir, "alpha_gamma_secp_pic4_{}.png".format(iters)))
    
if __name__ == "__main__":
    # dir = "/share/home/wuxingxing/datas/al_dir/cu_system/physical_character"
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_path', help='specify the output file', type=str, default='/share/home/wuxingxing/datas/al_dir/cu_system/physical_character')
    parser.add_argument('-t', '--iters', help='specify iters', type=int, default=1000000)
    parser.add_argument('-v', '--v0', help='specify v', type=float, default=0.9987)
    parser.add_argument('-l', '--lambda0', help='specify lambda', type=float, default=0.98)

    args = parser.parse_args()

    save_path = args.save_path
    iters = args.iters
    lambda0 = args.lambda0
    v0 = args.v0
    # save_path = dir
    # iters = 100
    calculate_params(save_path, iters, lambda0, v0)