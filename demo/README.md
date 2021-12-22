# Demo

- Front-end : streamlit
- Back-end : FastAPI

### 디렉토리 구조

```
/demo
    /lpips
    /data
        /actor_data
        561000_factorization.pt
        561000.pt
        566000.pt
        embedding_info.data
        embedding.data
    ...
```

### 명령어

```jsx
make -j 2 run_all 
```

> Makefile에 streamlit 서버 포트가 6006으로 설정되어 있습니다.
>
