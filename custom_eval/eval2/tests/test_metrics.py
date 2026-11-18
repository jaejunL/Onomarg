from eval2.metrics import score_file
def test_metrics(): 
 x=score_file(["a","a",""],["a","b","c"]); assert x["count"]==3 and x["empty_rate"]==1/3
