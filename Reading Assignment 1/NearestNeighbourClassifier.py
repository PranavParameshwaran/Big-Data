import math 
  
def classifyAPoint(points,p,k=3): 
    
    distance=[] 
    for group in points: 
        for feature in points[group]: 
            euclidean_distance = math.sqrt((feature[0]-p[0])**2 +(feature[1]-p[1])**2) 

            distance.append((euclidean_distance,group)) 
    distance = sorted(distance)[:k] 
  
    freq1 = 0 #frequency of group 0 
    freq2 = 0 #frequency og group 1 
  
    for d in distance: 
        if d[1] == 0: 
            freq1 += 1
        elif d[1] == 1: 
            freq2 += 1
  
    return 0 if freq1>freq2 else 1


def main(): 
    points = {0:[(1,1),(1,2),(2,3),(2,1)], 
              1:[(-1,1),(3,-1),(2,-1),(-2,0)]} 
  
    # testing point p(x,y) 
    p = (0,0) 
  
    # Number of neighbours  
    k = 5
  
    print("The value classified to unknown point is: {}".format(classifyAPoint(points,p,k))) 
  
if __name__ == '__main__': 
    main() 
      
