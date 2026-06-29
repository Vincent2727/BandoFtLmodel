import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

values = np.linspace(-2.1, 2.1, 100)

sum = 0
# trial 4
# list = [278.31, 251.45, 266.03, 231.97, 253.36, 354.52, 294.86, 223.83, 231.91, 297.04]

# Trial 7
# list = [159.95, 167.15, 164.44, 161.32, 163.38, 163.11, 162.2, 163.45, 165.15, 160.77, 165.31]

# Trial 7 (new)
list = [49.28, 48.46, 47.43, 47.86, 46.86, 47.62, 48.76, 48.72, 46.44, 44.2]

list = [104.94, 127.75, 114.81, 91.46, 111.0, 129.94, 133.11, 170.43, 172.31, 182.89, 168.2, 193.64, 181.53, 139.7, 97.39, 96.65, 63.8, 90.21, 47.24, 64.64]

# Trial 8
# list = [53.19, 44.73, 46.98, 44.81, 49.94, 52.45, 45.63, 46.06, 42.94, 46.2]
perturb = [-2.1,        -1.87894737, -1.65789474, -1.43684211, -1.21578947, -0.99473684,
 -0.77368421, -0.55263158, -0.33157895, -0.11052632,  0.11052632,  0.33157895,
  0.55263158,  0.77368421,  0.99473684,  1.21578947,  1.43684211,  1.65789474,
  1.87894737,  2.1,       ]

list = [166.86, 103.96, 128.09, 121.41, 104.51, 103.73, 127.1, 128.27, 168.13, 214.17, 223.15, 141.89, 221.56, 118.73, 133.08, 105.47, 112.17, 124.17, 94.83, 61.96]

list = [115.12, 110.13, 112.22, 181.21, 154.96, 162.56, 111.45, 236.63, 139.92, 230.44, 156.69, 143.99, 149.64, 127.44, 141.97, 126.53, 85.07, 67.16, 118.2, 100.89]

# initial position offset
list = [96.04, 105.88, 116.78, 128.94, 142.78, 159.0, 178.64, 203.77, 240.01, 313.32, 317.55, 252.64, 224.66, 207.51, 195.51, 186.45, 179.17, 173.03, 167.59, 162.57]

list = [31.24, 74.17, 112.91, 72.6, 112.7, 130.05, 112.04, 103.76, 190.9, 162.05, 206.65, 156.49, 181.82, 119.93, 128.46, 187.14, 78.1, 146.77, 143.17, 86.32]

step = 0

for i in list:
    print(f'Waves were observed starting at t={i}s, with an initial position perturbation of {perturb[step]}')
    sum += i
    step += 1

divisor = sum / len(list)
print(f"Average wave starting time: {round(divisor, 2)}\nStandard deviation: {round(np.std(list), 2)}")

# Plot the graph
plt.figure(figsize=(20, 7))
y = [51.54, 220.42, 69.19, 44.4, 91.6, 74.29, 98.12, 100.44, 80.79, 96.57, 102.05, 33.84, 102.49, 52.86, 137.69, 115.75, 78.64, 118.25, 147.48, 102.9, 104.21, 68.88, 70.24, 182.94, 81.51, 84.36, 69.97, 116.88, 101.2, 164.04, 85.56, 118.16, 58.9, 118.16, 170.11, 132.64, 162.09, 118.69, 138.59, 92.96, 125.2, 158.48, 208.73, 172.93, 139.16, 177.01, 149.93, 297.9, 251.7, 224.49, 188.95, 263.43, 191.67, 256.77, 206.45, 204.27, 202.15, 165.92, 157.7, 193.89, 153.99, 149.92, 120.64, 175.44, 117.74, 173.76, 123.65, 174.11, 136.09, 162.73, 206.86, 123.97, 93.65, 146.28, 130.77, 117.17, 164.03, 162.78, 135.41, 144.12, 154.74, 117.1, 136.75, 115.19, 113.36, 132.28, 166.77, 115.88, 128.07, 116.2, 111.75, 106.97, 94.82, 125.08, 72.12, 136.77, 95.28, 145.61, 96.39, 64.31]
plt.plot(values, y)

plt.title('Wave Started vs Velocity Perturbation')
plt.xlabel('Position Perturbation (m)')
plt.ylabel('Time Wave Started (s)')
plt.grid(True)
plt.tight_layout()
plt.ylim((30, 450))
# VelocityPerturbation_Accelerating_Values2.1
# VelocityPerturbation_Range2.1_No_Distractions
# VelocityPerturbation_Accelerating_Values2.1
# VelocityPerturbation_Constant_Speed_Values2.1
namef = f'/Users/vincentdegaetano/Documents/VsCodeProjects/Traffic_Images/Model_1_Images/PositionPerturbation_Distractions_Accelerating_Range2.1/WaveStartedvsPerturbation_{values[-1]}_Created_Distractions_Accelerating_Position.png'
plt.savefig(namef)



print(f'Saved Figure as {namef}')
