import numpy as np
import matplotlib.pyplot as plt
import csv

csv_path = 'dat/kawagoe_link_data.csv'
data = np.genfromtxt(csv_path, delimiter=',',skip_header=1)
# 基本となるデータ
data_kawagoe = np.delete(data, 0, axis=1)

# 1to3を進入禁止にしたデータ
data_kawagoe_not13 = data_kawagoe.copy()  # データのコピーを作成
for i in range(0,8):
   if i < 3:
        data_kawagoe_not13[i]+= data_kawagoe[3]
   elif i == 3:
        data_kawagoe_not13[i] = 0
        data_kawagoe_not13[i,14] = data_kawagoe[i,14] #0除算を避けるため
# 3to1を進入禁止にしたデータ
data_kawagoe_not31 = data_kawagoe.copy()  # データのコピーを作成
for i in range(0,8):
    if 4 <= i < 7:
        data_kawagoe_not31[i]+= data_kawagoe[7]
    elif i == 7:
          data_kawagoe_not31[i] = 0
          data_kawagoe_not31[i,14] = data_kawagoe[i,14] #0除算を避けるため

# 1to3と3to1を進入禁止にしたデータ
data_kawagoe_not13and31 = data_kawagoe.copy()  # データのコピーを作成
for i in range(0,8):
    if i < 3:
        data_kawagoe_not13and31[i]+= data_kawagoe[3]
    elif 4 <= i < 7:
        data_kawagoe_not13and31[i]+= data_kawagoe[7]
    elif i == 3 or i == 7:
        data_kawagoe_not13and31[i] = 0
        data_kawagoe_not13and31[i,14] = data_kawagoe[i,14] #0除算を避けるため

# 1to3と3to1を10~16時だけ進入禁止にしたデータ
data_kawagoe_not13and31_10to16 = data_kawagoe.copy()  # データのコピーを作成
for i in range(0,8):
    if i < 3:
        data_kawagoe_not13and31_10to16[i,3:9] += data_kawagoe[3,3:9]
    elif 4 <= i < 7:
        data_kawagoe_not13and31_10to16[i,3:9] += data_kawagoe[7,3:9]
    elif i == 3 or i == 7:
        data_kawagoe_not13and31_10to16[i,3:9] = 0
        data_kawagoe_not13and31_10to16[i,14] = data_kawagoe[i,14] #0除算を避けるため

# 1to3と3to1を10~17時だけ進入禁止にしたデータ
data_kawagoe_not13and31_10to17 = data_kawagoe.copy()  # データのコピーを作成
for i in range(0,8):
    if i < 3:
        data_kawagoe_not13and31_10to17[i,3:10] += data_kawagoe[3,3:10]
    elif 4 <= i < 7:
        data_kawagoe_not13and31_10to17[i,3:10] += data_kawagoe[7,3:10]
    elif i == 3 or i == 7:
        data_kawagoe_not13and31_10to17[i,3:10] = 0
        data_kawagoe_not13and31_10to17[i,14] = data_kawagoe[i,14] #0除算を避けるため

# print(data_kawagoe)
# print(data_kawagoe_not13and31)
# print(data_kawagoe_not13and31_10to16)


# aとρの導出(t=a+ρb)
listrho = []
lista = []
for i in range(0,8):
    #print(np.floor(data_kawagoe[i]))
    len_load = float(data_kawagoe[i,12])
    ave_speed = float(data_kawagoe[i,13])/10
    speed_limit = float(data_kawagoe[i,14])
    sum_traffic = 0

    for j in range(0,12):
     sum_traffic += float(data_kawagoe[i,j])
    #print(sum_traffic)

    a = len_load/speed_limit
    rho = (12.0*(len_load/ave_speed)-12.0*a)/sum_traffic
    listrho.append(rho)
    lista.append(a)
# 24の傾きを是正した後のρ
listrho_dash = listrho.copy()
listrho_dash[1] = listrho[3]  # 24の傾きを是正
listrho_dash[5] = listrho[7]  # 42の傾きを是正

print("rho:", listrho)
print("rhodash:", listrho_dash)
print("a:", lista)

# 各リンクの時間を計算
def calculatetime(X,listx):
    cal_time = np.zeros((1,12))
    for i in range(0,8):
        cal_time2=[]
        for j in range(0,12):
            cal_time2.append(float(X[i,12]/X[i,14] + listx[i]*X[i,j]))
        cal_time3=np.array([cal_time2])
        cal_time=np.concatenate([cal_time,cal_time3],axis=0)
    return np.delete(cal_time, 0, axis=0)*0.001*60 # 時間を分に変換
    
# 各手法の時間を計算
time = calculatetime(data_kawagoe,listrho)
time_not13 = calculatetime(data_kawagoe_not13,listrho)
time_not31 = calculatetime(data_kawagoe_not31,listrho)
time_not13and31 = calculatetime(data_kawagoe_not13and31,listrho)
time_not13and31_10to16 = calculatetime(data_kawagoe_not13and31_10to16,listrho)
time_not13and31_10to17 = calculatetime(data_kawagoe_not13and31_10to17,listrho)
# 各手法の時間を計算（24の傾きを是正後）
time_dash = calculatetime(data_kawagoe,listrho_dash)
time_not13_dash = calculatetime(data_kawagoe_not13,listrho_dash)
time_not31_dash = calculatetime(data_kawagoe_not31,listrho_dash)
time_not13and31_dash = calculatetime(data_kawagoe_not13and31,listrho_dash)
time_not13and31_10to16_dash = calculatetime(data_kawagoe_not13and31_10to16,listrho_dash)
time_not13and31_10to17_dash = calculatetime(data_kawagoe_not13and31_10to17,listrho_dash)
# グラフの描画
x = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

for i in range(0,8):
    if 0 <= i < 3:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not13[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time','time_not13'])
        if i == 0:
            fig.savefig(f'out/time_12.png')
        elif i == 1:
            fig.savefig(f'out/time_24.png')
        elif i == 2:
            fig.savefig(f'out/time_43.png')
        plt.close(fig)
    elif 3 < i < 8:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not31[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time','time_not31'])
        if i == 4:
            fig.savefig(f'out/time_34.png')
        elif i == 5:
            fig.savefig(f'out/time_42.png')
        elif i == 6:
            fig.savefig(f'out/time_21.png')
        plt.close(fig)
#歩行者天国で10~16時まで時間制限をした時のグラフ
for i in range(0,8):
    if 0 <= i < 3:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not13and31_10to16[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time_not13and31_10to16','time_not13and31_10to16_dash'])
        if i == 0:
            fig.savefig(f'out/time_12_10to16.png')
        elif i == 1:
            fig.savefig(f'out/time_24_10to16.png')
        elif i == 2:
            fig.savefig(f'out/time_43_10to16.png')
        plt.close(fig)
    elif 3 < i < 8:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not13and31_10to16[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time_not13and31_10to16','time_not13and31_10to16_dash'])
        if i == 4:
            fig.savefig(f'out/time_34_10to16.png')
        elif i == 5:
            fig.savefig(f'out/time_42_10to16.png')
        elif i == 6:
            fig.savefig(f'out/time_21_10to16.png')
        plt.close(fig)
# 歩行者天国で10~17時まで時間制限をした時のグラフ
for i in range(0,8):
    if 0 <= i < 3:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not13and31_10to17[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time_not13and31_10to17','time_not13and31_10to17_dash'])
        if i == 0:
            fig.savefig(f'out/time_12_10to17.png')
        elif i == 1:
            fig.savefig(f'out/time_24_10to17.png')
        elif i == 2:
            fig.savefig(f'out/time_43_10to17.png')
        plt.close(fig)
    elif 3 < i < 8:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x, time[i], "-", c="Blue", linewidth=1, marker='o', alpha=1)
        ax.plot(x, time_not13and31_10to17[i], "-", c="Red", linewidth=1, marker='o', alpha=1)
        ax.legend(['time_not13and31_10to17','time_not13and31_10to17_dash'])
        if i == 4:
            fig.savefig(f'out/time_34_10to17.png')
        elif i == 5:
            fig.savefig(f'out/time_42_10to17.png')
        elif i == 6:
            fig.savefig(f'out/time_21_10to17.png')
        plt.close(fig)

# 24の傾きを是正した場合のグラフ
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, time_dash[3], "-", c="Blue", linewidth=1, marker='o', alpha=1)
ax.plot(x, time_not13_dash[3], "-", c="Red", linewidth=1, marker='o', alpha=1)
ax.legend(['time_dash','time_not13_dash'])
fig.savefig(f'out/time_24_dash.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, time_dash[5], "-", c="Blue", linewidth=1, marker='o', alpha=1)
ax.plot(x, time_not31_dash[5], "-", c="Red", linewidth=1, marker='o', alpha=1)
ax.legend(['time_dash','time_not31_dash'])
fig.savefig(f'out/time_42_dash.png')
plt.close(fig)

# 旅行時間の増加量(一次元配列)
def calculate_increase_time(time_increase,data_kawagoe_increase):
    increase_time = np.zeros((8,12))
    for i in range(0,8):
        for j in range(0,12):
            increase_time[i,j] = (time_increase[i,j]*data_kawagoe_increase[i,j] - time[i,j]*data_kawagoe[i,j])/60 #時間に戻す
    return(increase_time)

# print("not13,",np.sum(calculate_increase_time(time_not13,data_kawagoe_not13)))
# print("not31,",np.sum(calculate_increase_time(time_not31,data_kawagoe_not31)))
# print("not13&31,",np.sum(calculate_increase_time(time_not13and31,data_kawagoe_not13and31)))
# # 24の傾きを是正した場合の増加量
# print("timedash,",np.sum(calculate_increase_time(time_dash,data_kawagoe)))
# print("not13dash,",np.sum(calculate_increase_time(time_not13_dash,data_kawagoe_not13)))
# print("not31dash,",np.sum(calculate_increase_time(time_not31_dash,data_kawagoe_not31)))
# print("not13&31dash,",np.sum(calculate_increase_time(time_not13and31_dash,data_kawagoe_not13and31)))
# # 歩行者天国の時間を10~16時に限定したときの増加量
# print("not13&31_10to16,",np.sum(calculate_increase_time(time_not13and31_10to16,data_kawagoe_not13and31_10to16)))
# print("not13&31_10to16_dash,",np.sum(calculate_increase_time(time_not13and31_10to16_dash,data_kawagoe_not13and31_10to16)))
# 歩行者天国の時間を10~17時に限定したときの増加量
print("not13&31_10to17,",np.sum(calculate_increase_time(time_not13and31_10to17,data_kawagoe_not13and31_10to17)))
print("not13&31_10to17_dash,",np.sum(calculate_increase_time(time_not13and31_10to17_dash,data_kawagoe_not13and31_10to17)))

# 旅行時間の合計
def calculate_total_time(time_total,data_kawagoe_total):
    traffic = np.zeros((8,12))
    for i in range(0,8):
        for j in range(0,12):
            traffic[i,j] = time_total[i,j]*data_kawagoe_total[i,j]/60
    return np.sum(traffic)
print("time,",calculate_total_time(time,data_kawagoe))
print("time_not13,",calculate_total_time(time_not13,data_kawagoe_not13))
print("time_not31,",calculate_total_time(time_not31,data_kawagoe_not31))
print("time_not13and31,",calculate_total_time(time_not13and31,data_kawagoe_not13and31))
print("timedash,",calculate_total_time(time_dash,data_kawagoe))
print("not13dash,",calculate_total_time(time_not13_dash,data_kawagoe_not13))
print("not31dash,",calculate_total_time(time_not31_dash,data_kawagoe_not31))
print("not13&31dash,",calculate_total_time(time_not13and31_dash,data_kawagoe_not13and31))
print("not13&31_10to16,",calculate_total_time(time_not13and31_10to16,data_kawagoe_not13and31_10to16))
print("not13&31_10to16_dash,",calculate_total_time(time_not13and31_10to16_dash,data_kawagoe_not13and31_10to16))
print("not13&31_10to17,",calculate_total_time(time_not13and31_10to17,data_kawagoe_not13and31_10to17))
print("not13&31_10to17_dash,",calculate_total_time(time_not13and31_10to17_dash,data_kawagoe_not13and31_10to17))

# CSVファイルへの出力
outcsv=np.array(['time_12(m)', 'time_24', 'time_43', 'time_13', 'time_34', 'time_42', 'time_21', 'time_31'])
out_time = np.insert(time.astype(str), 0, outcsv, axis=1)
out_time_not13 = np.insert(time_not13.astype(str), 0, outcsv, axis=1)
out_time_not31 = np.insert(time_not31.astype(str), 0, outcsv, axis=1)
out_time_not13and31 = np.insert(time_not13and31.astype(str), 0, outcsv, axis=1)
out_time_not13and31_10to16 = np.insert(time_not13and31_10to16.astype(str), 0, outcsv, axis=1)
# 24の傾きを是正した場合のCSVファイルへの出力
out_time_dash = np.insert(time_dash.astype(str), 0, outcsv, axis=1)
out_time_not13_dash = np.insert(time_not13_dash.astype(str), 0, outcsv, axis=1)
out_time_not31_dash = np.insert(time_not31_dash.astype(str), 0, outcsv, axis=1)
out_time_not13and31_dash = np.insert(time_not13and31_dash.astype(str), 0, outcsv, axis=1)
out_time_not13and31_10to16_dash = np.insert(time_not13and31_10to16_dash.astype(str), 0, outcsv, axis=1)
out_time_not13and31_10to17 = np.insert(time_not13and31_10to17.astype(str), 0, outcsv, axis=1)
out_time_not13and31_10to17_dash = np.insert(time_not13and31_10to17_dash.astype(str), 0, outcsv, axis=1)
header = ['path_timeperiod',7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

def write_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

write_csv('out/kawagoe_base_out.csv', out_time)
write_csv('out/kawagoe_not13_out.csv', out_time_not13)
write_csv('out/kawagoe_not31_out.csv', out_time_not31)
write_csv('out/kawagoe_not13and31_out.csv', out_time_not13and31)
write_csv('out/kawagoe_base_out_dash.csv', out_time_dash)
write_csv('out/kawagoe_not13_out_dash.csv', out_time_not13_dash)
write_csv('out/kawagoe_not31_out_dash.csv', out_time_not31_dash)
write_csv('out/kawagoe_not13and31_out_dash.csv', out_time_not13and31_dash)
write_csv('out/kawagoe_not13and31_10to16_out.csv', out_time_not13and31_10to16)
write_csv('out/kawagoe_not13and31_10to16_out_dash.csv', out_time_not13and31_10to16_dash)
write_csv('out/kawagoe_not13and31_10to17_out.csv', out_time_not13and31_10to17)
write_csv('out/kawagoe_not13and31_10to17_out_dash.csv', out_time_not13and31_10to17_dash)
