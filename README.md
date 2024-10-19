# VIT

将CIFAR10_balance 和 CIFAR10_imbalanced 数据集放在根目录下，如下所示
VIT
├─CIFAR10_balance
│  ├─0
│  ├─1
│  ├─2
│  ├─3
│  ├─4
│  ├─5
│  ├─6
│  ├─7
│  ├─8
│  └─9
├─CIFAR10_imbalanced
│  ├─0
│  ├─1
│  ├─2
│  ├─3
│  ├─4
│  ├─5
│  ├─6
│  ├─7
│  ├─8
│  └─9
├─README.md
├─...

运行train_oversample以利用过采样方式类均衡并训练模型
运行train_weightloss以利用权重损失方式类均衡并训练模型
运行test以测试模型（记得改开头的model_pth_path）