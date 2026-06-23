import asyncio
import pandas as pd
from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

# 读取数据
ratings_data = pd.read_csv("M_ML-100K/ratings.dat", sep="::", encoding='latin', header=None, engine='python')
ratings_data.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
# print(ratings_data.head())
movies_data = pd.read_csv("M_ML-100K/movies.dat", sep="::", encoding='latin', header=None, engine='python')
movies_data.columns = ['movie_id', 'title', 'tag']
movies_map = movies_data.set_index('movie_id')['title'].to_dict()
movies = movies_data['title'].tolist()
# print(movies_data.head())
# print(movies_map)
print(movies)

# 用户观看所有的电影记录
target_user_id = 196
target_ratings = ratings_data.loc[ratings_data['user_id'] == target_user_id]
target_ratings = target_ratings.loc[target_ratings['rating'] >= 3]
target_ratings_len = len(target_ratings)
# print(target_ratings)
print("历史记录条数：", target_ratings_len)

# 拆分数据集 比例2:1
# 已观看的电影记录
had_ratings = target_ratings.iloc[:int(target_ratings_len/3*2)]
print("已观看记录：\n", had_ratings)
# 即将观看的电影记录
will_ratings = target_ratings.iloc[int(target_ratings_len/3*2):]
print("即将观看记录：\n", will_ratings)

# 已观看的电影记录
had_movie_id_list = had_ratings['movie_id'].tolist()
had_movie_title_list = [movies_map[id] for id in had_movie_id_list]
# print(had_movie_id_list)
# print(had_movie_title_list)
# 即将观看的电影记录
will_movie_id_list = will_ratings['movie_id'].tolist()
will_movie_title_list = [movies_map[id] for id in will_movie_id_list]
# print(had_movie_id_list)
# print(had_movie_title_list)

# 大模型 没有微调
external_client = AsyncOpenAI(
    api_key="sk-b95acfe4573e485cbd51b6bb3f3badd4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
create_prompt = f"""
你是一个电影推荐专家，请根据用户观看电影的历史记录，推测30个用户未来可能观看的电影名称。
输出格式：不要解释，不要其他输出，每一行是一个推荐。
"""
create_agent = Agent(
    "电影推荐专家",
    instructions=create_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    )
)


async def main():
    result = await create_search(had_movie_title_list)
    print(result)
    create_result = result.split("\n")
    print("生成搜索结果->", create_result)
    result = await bm_search(create_result, movies)
    print("bm25检索结果->", result)
    hit = 0
    for item in result:
        if item in will_movie_title_list:
            hit += 1
    print("准确率：", hit / len(result))

async def create_search(movie_list):
    """生成电影搜索查询"""
    history_movie_text = f"""
    用户观看电影的历史记录：
{"\n".join(movie_list)}
"""
    result = await Runner.run(
        create_agent,
        history_movie_text
    )
    print(result)
    return result.final_output


async def bm_search(create_result, all_movies):
    """BM25检索电影结果"""
    tokenized_corpus = [item.lower().split() for item in all_movies]
    bm25 = BM25Okapi(tokenized_corpus)
    movie_score = dict()
    for item in create_result:
        scores = bm25.get_scores(item.lower().split())
        # 每次取
        top_n = 5
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        print("item score", "-" * 100)
        for idx in top_indices:
            print(f"{all_movies[idx]} | 得分：{scores[idx]:.4f}")
            movie = all_movies[idx]
            if movie in movie_score:
                movie_score[movie] += scores[idx]
            else:
                movie_score[movie] = scores[idx]
    # 最后得到10个结果
    top_n = 10
    last_result = sorted(
        movie_score.items(),
        key=lambda item: item[1],
        reverse=True
    )[:top_n]
    print("last score", "-" * 100)
    results = list()
    for item in last_result:
        results.append(item[0])
        print(f"{item[0]} | 得分：{item[1]:.4f}")
    return results


if __name__ == '__main__':
    asyncio.run(main())


