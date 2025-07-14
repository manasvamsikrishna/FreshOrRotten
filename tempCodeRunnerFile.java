class Solution {

    public static int towerOfHanoi(int n, int from, int to, int aux) {
        // Your code here
        int count=0;
        
        if(n==1){
            count++;
            
        }
        towerOfHanoi(n-1,from,aux,to);
        towerOfHanoi(n-1,aux,to,from);
        
        return count;
    }
    public static void main(String args[]){
        int n=3;
        int from=3;
        int to=0;
        int aux=0;
        System.out.println(towerOfHanoi(n,from,to,aux));
    }
}
