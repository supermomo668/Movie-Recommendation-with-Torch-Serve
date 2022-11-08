export model_name=mar
export test_user_id=1   # 0 to 67786
curl -X POST http://localhost:8080/predictions/$model_name -H 'Content-Type: application/json' -d "{'user_id':$test_user_id}"