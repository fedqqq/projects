import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI(title="Recommendation Service")

state = None
articles_df = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]


def load_models():
    global state, articles_df
    state = joblib.load('model_state.joblib')
    articles_df = pd.read_csv('articles.csv')
    articles_df['contentId'] = articles_df['contentId'].astype(int)
    print("Models loaded")


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/")
def root():
    return {"message": "Recommendation service is running. Use /docs for API documentation or /recommend?user_id=..."}


def get_als_scores(user_id):
    if user_id not in state['als']['user_to_idx']:
        return None
    u_idx = state['als']['user_to_idx'][user_id]
    user_vec = state['als']['user_factors'][u_idx]
    scores = np.dot(state['als']['item_factors'], user_vec)
    return {state['als']['idx_to_item'][i]: float(scores[i]) for i in range(len(scores))}


def get_ranker_scores(user_id, candidate_items):
    if user_id not in state['ranker']['user_to_idx']:
        return None
    u_idx = state['ranker']['user_to_idx'][user_id]
    user_vec = state['ranker']['user_emb'][u_idx]
    scores = {}
    for cid in candidate_items:
        if cid in state['ranker']['item_to_idx']:
            i_idx = state['ranker']['item_to_idx'][cid]
            item_vec = state['ranker']['item_emb'][i_idx]
            scores[cid] = float(np.dot(user_vec, item_vec))
    return scores


def recommend_items(user_id, topn=10, exclude_interacted=None):
    if exclude_interacted is None:
        exclude_interacted = []

    als_scores = get_als_scores(user_id)
    if als_scores is None:
        pop = state['als']['popular_items']
        candidates = [i for i in pop if i not in exclude_interacted][:topn]
        return pd.DataFrame(candidates, columns=['contentId'])

    sorted_als = sorted(als_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_items = [cid for cid, _ in sorted_als if cid not in exclude_interacted][:200]

    ranker_scores = get_ranker_scores(user_id, candidate_items)
    if ranker_scores is None:
        final_scores = {cid: als_scores.get(cid, 0) for cid in candidate_items}
    else:
        final_scores = ranker_scores

    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    recommendations = [cid for cid, _ in sorted_items]

    return pd.DataFrame(recommendations, columns=['contentId'])


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
        user_id: int = Query(..., description="User ID"),
        topn: int = Query(10, ge=1, le=100)
):
    try:
        recs_df = recommend_items(user_id, topn=topn)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    recs_df = recs_df.merge(articles_df[['contentId', 'title', 'url']], on='contentId', how='left')
    recommendations = recs_df[['contentId', 'title', 'url']].to_dict(orient='records')
    return RecommendationResponse(user_id=user_id, recommendations=recommendations)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)