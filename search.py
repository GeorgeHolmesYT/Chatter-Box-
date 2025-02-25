from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from redis import Redis

class SearchEngine:
    def __init__(self, elastic_url: str, redis_url: str):
        self.es = AsyncElasticsearch([elastic_url])
        self.redis = Redis.from_url(redis_url)
        self.vectorizer = TfidfVectorizer()
        
    async def index_message(self, message: Dict):
        await self.es.index(
            index="messages",
            document={
                "content": message["content"],
                "userId": message["userId"],
                "roomId": message["roomId"],
                "timestamp": datetime.now(),
                "messageType": message["type"],
                "metadata": message.get("metadata", {})
            }
        )
        
    async def search_messages(self, query: str, filters: Dict = None) -> List[Dict]:
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": query}}
                    ]
                }
            },
            "sort": [{"timestamp": "desc"}],
            "size": 50
        }
        
        if filters:
            search_body["query"]["bool"]["filter"] = [
                {"term": {k: v}} for k, v in filters.items()
            ]
            
        results = await self.es.search(index="messages", body=search_body)
        return [hit["_source"] for hit in results["hits"]["hits"]]
        
    async def semantic_search(self, query: str, context: str) -> List[Dict]:
        query_vector = self.vectorizer.fit_transform([query]).toarray()[0]
        
        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['content_vector']) + 1.0",
                        "params": {"query_vector": query_vector.tolist()}
                    }
                }
            },
            "size": 20
        }
        
        results = await self.es.search(index="messages", body=search_body)
        return [hit["_source"] for hit in results["hits"]["hits"]]
        
    async def index_user(self, user: Dict):
        await self.es.index(
            index="users",
            document={
                "userId": user["id"],
                "username": user["username"],
                "email": user["email"],
                "metadata": user.get("metadata", {})
            }
        )
        
    async def search_users(self, query: str) -> List[Dict]:
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["username^2", "email"]
                }
            }
        }
        
        results = await self.es.search(index="users", body=search_body)
        return [hit["_source"] for hit in results["hits"]["hits"]]
        
    async def index_room(self, room: Dict):
        await self.es.index(
            index="rooms",
            document={
                "roomId": room["id"],
                "name": room["name"],
                "description": room.get("description", ""),
                "members": room["members"],
                "metadata": room.get("metadata", {})
            }
        )
        
    async def search_rooms(self, query: str, user_id: str) -> List[Dict]:
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"name": query}},
                        {"term": {"members": user_id}}
                    ]
                }
            }
        }
        
        results = await self.es.search(index="rooms", body=search_body)
        return [hit["_source"] for hit in results["hits"]["hits"]]
        
    def cache_search_results(self, query: str, results: List[Dict]):
        self.redis.setex(
            f"search:{query}",
            300,  # Cache for 5 minutes
            str(results)
        )
        
    def get_cached_results(self, query: str) -> Optional[List[Dict]]:
        cached = self.redis.get(f"search:{query}")
        return eval(cached) if cached else None
