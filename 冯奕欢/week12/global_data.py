from agents import SQLiteSession

session_map = dict()


def get_session(user):
    """
    获取用户的会话缓存
    :param user: 用户
    :return: 目标用户的会话缓存
    """
    if user in session_map:
        return session_map[user]
    session = SQLiteSession(f"chat_ai_{user}")
    session_map[user] = session
    return session


def reset_session(user):
    """
    重置用户的会话缓存
    :param user: 用户
    :return: 无
    """
    session = SQLiteSession(f"chat_ai_{user}")
    session_map[user] = session
