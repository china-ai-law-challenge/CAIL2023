1. **torch1.11.md** 中包含了评测使用的具体环境信息，如果选择 **torch1.11** 环境，选手使用的代码需要能在该环境下运行。

2. **requirements.txt** 文件中包含了构建该环境时使用的依赖包。**torch1.11**由下面的命令安装。

   ```
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

   **paddle** 由下面的命令安装。
   
   ```
   python3 -m pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
   ```
   
   
