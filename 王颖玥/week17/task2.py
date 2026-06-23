"""
GPT4Rec ：基于生成式语言模型的个性化推荐框架，借助 GPT2 生成未来的查询条件，使用搜索检索到相关的物品。
- 步骤1（生成查询条件）: 根据用户历史交互物品的文本信息（如商品标题），生成能够代表用户未来兴趣的、可读的“搜索查询”。
    Previously, the customer has bought: <标题1>. <标题2>... In the future, the customer wants to buy
- 步骤2（物品的检索）: 从整个物品库中检索出最相关的物品作为推荐候选
"""

import pandas as pd
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-078ae61448344f53b3cb03bcc85ff7cd"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请严格按照以下要求输出：
1. 仅基于用户历史观看的电影，推荐用户未来可能观看的10部电影；
2. 输出格式为：每一行仅包含电影名和年份，不要输出其他文字；
3. 禁止输出任何解释等非电影标题和年份的内容

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个候选的电影描述，每一行是一个推荐。
"""


def load_data():
    ratings = pd.read_csv(
        "./M_ML-100K/ratings.dat",
        sep="::",
        header=None,
        engine='python'
    )
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

    movies = pd.read_csv(
        "./M_ML-100K/movies.dat",
        sep="::",
        header=None,
        engine='python',
        encoding='latin-1'
    )
    movies.columns = ["movie_id", "movie_title", "movie_tag"]

    return ratings, movies


def get_user_history(user_id, ratings, movies):
    if user_id not in ratings["user_id"].unique():
        raise ValueError(
            f"用户 {user_id} 不存在，请重新输入！（用户ID有效范围为 {ratings["user_id"].min()}-{ratings["user_id"].max()}）"
        )

    user_ratings = ratings[ratings["user_id"] == user_id]
    user_history = pd.merge(
        user_ratings, movies, on="movie_id", how="left"
    )[["movie_title", "rating"]].sort_values(by="rating", ascending=False)

    user_history_titles = user_history["movie_title"].tolist()[:20]
    if len(user_history_titles) == 0:
        raise ValueError(f"用户 {user_id} 没有观影记录！")

    return user_history_titles, user_ratings


# 选取最终的推荐电影
def final_recommend(candidates, user_ratings, ratings, movies):
    # 1、获取用户已看电影标题
    watched_ids = user_ratings["movie_id"].tolist()
    watched_titles = movies[movies["movie_id"].isin(watched_ids)]["movie_title"].tolist()

    # 2、过滤已看电影
    filtered_movies = [title for title in candidates if title not in watched_titles]
    if not filtered_movies:
        filtered_movies = candidates

    # 3、计算每个电影的平均分
    title_to_score = {}
    movie_avg_score = ratings.groupby("movie_id")["rating"].mean()
    for i in range(len(movies)):
        movie_title = movies.iloc[i]["movie_title"]
        movie_id = movies.iloc[i]["movie_id"]
        avg_score = movie_avg_score.get(movie_id, 0)
        title_to_score[movie_title] = avg_score

    # 给每个候选电影匹配相应的评分
    candidates_with_score = []
    for title in filtered_movies:
        score = title_to_score.get(title, 0)
        candidates_with_score.append((title, score))
    # 按评分降序排序
    candidates_with_score.sort(key=lambda x: x[1], reverse=True)

    # 选评分最高的那个
    final_recommend = candidates_with_score[0][0]
    final_rating = candidates_with_score[0][1] if candidates_with_score[0][1] != 0 else -1

    return final_recommend, final_rating


def main(user_id):
    ratings, movies = load_data()
    user_history_movies, user_ratings = get_user_history(user_id, ratings, movies)

    print(f"\n用户 {user_id} 观影历史，按评分降序排列：")
    for i, movie in enumerate(user_history_movies):
        print(f"{i + 1}: {movie}")

    # 调用大模型得到十个候选电影
    prompt = PROMPT_TEMPLATE.format("\n".join(user_history_movies))
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )

    try:
        response = client.chat.completions.create(
            model='qwen-max',
            messages=[
                {"role": "system", "content": "你是一个电影推荐专家"},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        candidates = [line.strip() for line in response_text.split("\n") if line.strip()]
    except Exception as e:
        raise Exception(f"调用失败：{str(e)}")

    print("\n十个候选电影为：")
    for i, candidate in enumerate(candidates):
        print(f"{i + 1}: {candidate}")

    # 从十个候选电影中过滤掉用户已经看过的，按照平均分降序排序，取评分最高的电影
    final_recommend_movie, final_rating = final_recommend(candidates, user_ratings, ratings, movies)
    return final_recommend_movie, final_rating


if __name__ == "__main__":
    user_id = int(input("请输入用户ID："))
    try:
        final_recommend_movie, final_rating = main(user_id)
        if final_rating == -1:
            print("\n电影无评分！")
        else:
            print(f"\n最终推荐电影为：{final_recommend_movie}，评分为：{final_rating}")
    except Exception as e:
            print(f"电影推荐失败：{str(e)}")


