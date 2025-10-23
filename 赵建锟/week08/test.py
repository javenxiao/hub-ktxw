import pytest
from fastapi.testclient import TestClient
from zuoye import app

client = TestClient(app)


class TestApp:


    def test_health_check(self):

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        print(data)
        assert data["status"] == "healthy"

    def test_extract_endpoint_success(self):

        test_data = {
            "text": "播放周杰伦的青花瓷",
            "model": "deepseek-chat"
        }
        response = client.post("/extract", json=test_data)


        assert response.status_code == 200


        data = response.json()
        print(data)
        assert "domain" in data
        assert "intent" in data
        assert "slots" in data
        assert isinstance(data["slots"], dict)

    def test_extract_endpoint_empty_text(self):

        test_data = {
            "text": "",
            "model": "deepseek-chat"
        }
        response = client.post("/extract", json=test_data)

        assert response.status_code == 422

    def test_coze_webhook(self):

        test_data = {
            "message": "查询天气",
            "user_id": "test_user_123"
        }
        response = client.post("/coze/webhook", json=test_data)
        assert response.status_code == 200
        data = response.json()
        print(data)
        assert "reply" in data
        assert "success" in data


@pytest.mark.asyncio
async def test_async_example():
    assert True