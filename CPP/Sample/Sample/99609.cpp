// hello Ryan
// non negative float number Sqrt()

/*
1. exp / 2 (perhaps a -1 normalization later)
2. actual number

64 => 1, 4, 9, ... [sum of odd numbers... 1+3+5+7...]

64 => (64/[2..32])^2 ... (2,3...32) log2(64)

*/


float sqrt(float x) 
{
    float start, end, root, delta;
    
    // range is [0 to x/2]
    start = 0.0;
    end = x/2;
    root = 0.0;
    delta = 0.0;

    // binary search the space [start to end] to find y such that y*y => x
    do {
        root = (start + end) / 2;    // TODO : Assert start + end will never overflow
        delta = x - (root*root);

        if ((delta < epsilon) && (-delta < epsilon)) break;
                
        if (delta < 0.0) // too large, go down
            end = root;
        else
            start = root;
 
    } while (start <= end);
    
    return root;
}