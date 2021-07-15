# TalkBot

> ### Hi,TalkBot!

TalkBot是一个融合怪ChatBot,主要用的是seq2seq生成对话

生成对话是源自ChatBotCourse

## 在开始之前

### 安装依赖

应为本项目使用的是poetry作为包相关管理工具,so,你只需要执行poetry补全依赖即可

### 处理未处理过的语料库

花了一点时间写了一个小工具,其包名为`TalkBot.read_crops`

可以这样使用

```python
from TalkBot.reader_crops import crops_reader

result = crops_reader(["/path/to/预料1.txt","/path/to/预料2.txt"],"./TalkBot/crops")

print(result)
```

如果没有错误之类的会输出

```
READ SUCCESSFUL
```

(不过啦,是有内置的语料库的)

### 更改相关配置

TalkBot的配置文件在`TalkBot/config.yml`中

其结构为

```yaml
Bot:
  Name: "ChiBot"
  Local: "深圳"

Wording:
  #不想回答
  dwta: 
    - "喂喂喂,你烦不烦啊"
    - "***,你好烦哦"
    
API:
  weather: "1234"
```

其中需要修改Name Local dwta 和相关的APIKey

dwta可能需要两个以上

### 训练AI模型

## 开始使用

只需引用`TalkBot`这个包即可

```python
import TalkBot

result = TalkBot.Talk(UserID=1234,Message="巴拉巴拉巴拉")

print(result)
```

## Other

这里要感谢ChatBotCourse提供了一个蛮好的使用seq2seq网络去生成对话的ChatBot教程,应为有了此教程,TalkBot才会如此简单的开发出来

如果有Bug,可以提交issue

如果有新function想合并到TalkBot欢迎pr!

