1. **torch1.13.md** 中包含了评测使用的具体环境信息，如果选择 **torch1.13** 环境，选手使用的代码需要能在该环境下运行。

2. **requirements.txt** 文件中包含了构建该环境时使用的依赖包。**torch1.13** 由下面的命令安装。

   ```
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

   **paddle** 由下面的命令安装。
   
   ```
   python -m pip install paddlepaddle-gpu==2.5.0.post116
   ```
   
   
