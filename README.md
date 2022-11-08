# Movie-Recommendation-with-Torch-Serve
I4 - ML Ops Tool for 17645 

Supplement full implementation based on the Medium blogpost: <br> https://medium.com/@mymomo119966.mm/ml-ops-with-containerized-torch-serve-movie-recommendation-as-example-b96663b4131c </br>

# Running Inference 
Run the inference upon deploying the model in the following manners!
```
curl -X POST http://localhost:8080/predictions/movie_recommender -H 'Content-Type: application/json' -d '{"user_id": 1}'
```
or run bash script ```test_inference.sh``` (command contained)
<br>
Our console returned:
```
[
 'there+will+be+blood+2007',
 'legends+of+the+fall+1994',
 'the+dark+knight+2008',
 'the+shawshank+redemption+1994',
 'the+silence+of+the+lambs+1991',
]
```
# Further

* Substitute your own ***model/model.py*** and ***model/handler.py*** and produce any inference endpoint!
