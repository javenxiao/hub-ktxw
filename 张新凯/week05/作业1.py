from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(
    api_key="<DeepSeek API Key>",
    base_url="https://api.deepseek.com"
)

prompt = """你是一个专业文本分析专家，请帮我对如下的文本进行分类：
还有双鸭山到淮阴的汽车票吗13号的

可以参考的样本如下（假如已有的训练集）：
查询北京飞桂林的飞机是否已经起飞了	Travel-Query
从这里怎么回家	Travel-Query
随便播放一首专辑阁楼里的佛里的歌	Music-Play
给看一下墓王之王嘛	FilmTele-Play
我想看挑战两把s686打突变团竞的游戏视频	Video-Play
我想看和平精英上战神必备技巧的游戏视频	Video-Play

你只能从如下的类别选择：['FilmTele-Play', 'Video-Play', 'Music-Play', 'Radio-Listen',
       'Alarm-Update', 'Weather-Query', 'Travel-Query',
       'HomeAppliance-Control', 'Calendar-Query', 'TVProgram-Play',
       'Audio-Play', 'Other']

只需要输出结果，不需要额外的解释。
"""

stream_switch = False
completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=stream_switch
)
if stream_switch:
    for chunk in completion:
        print(chunk.model_dump_json())
else:
    print(completion.model_dump_json())
