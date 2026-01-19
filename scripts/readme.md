# 程序入口

- configs：各个张量的配置文件
- read_json.ipynb:读取并处理各个张量使其与弹性张量结构一致
- run_eval_test_mae.sh：对训练好的模型进行测试，并计算张量的MAE
- submit_di.sh：介电张量的运行脚本
- submit_piezo.sh:压电张量的运行脚本
- train_materials_tensor_dielectric/piezo：介电/压电张量运行的python主程序