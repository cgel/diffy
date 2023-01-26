#include "./net.cpp"
#include <string.h>
#include <chrono>
using namespace std::chrono;



void NN_perf() {
    int hidden_dim = 256;
    int input_dim = 28*28;
	cout << "Perf of 3 layer NN with input dim " << input_dim << " hidden dim " << hidden_dim << " and output dim 1\n";

	Mat<float> A(hidden_dim, input_dim), B(hidden_dim,hidden_dim), C(1,hidden_dim);
    A.random(0, (2./input_dim)); B.random(0, (2./hidden_dim)); C.random(0, (2./hidden_dim));
    
    Mat<float> Lx, LA, LB, LC, loss;
    int num = 1;
    Mat<float> img(28*28,1);
	cout << "Average time of forward pass:\n";
    auto start = high_resolution_clock::now();
    for (int t=0; t<num; t++) {
	    Mat<float> y = nn.at(img,A,B,C);
        cout << t << " - " << y.ind(0,0) << endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start)/num;
    cout << duration.count() << " milliseconds\n";

	cout << "Average time to compute gradients (forward mode):\n";
    start = high_resolution_clock::now();
    for (int t=0; t<num; t++) {
        tie(Lx, LA, LB, LC) = nn.differential_at(img,A,B,C);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start)/num;
    cout << duration.count() << " milliseconds\n";
}

void matmul_perf() {
    Mat<float> A(1000,1000);
    Mat<float> B(1000,1000);
	cout << "Time to multiply two 1000x1000 empty matrices\n";
    auto start = high_resolution_clock::now();
    Mat<float> C = A*B;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " milliseconds\n";

	cout << "Time to fill a 1000x1000 uniform random matrix\n";
    start = high_resolution_clock::now();
    A.random(0,1);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " milliseconds\n";
    B.random(0,1);
	cout << "Time to multiply two 1000x1000 uniform random matrices\n";
    start = high_resolution_clock::now();
    C = A*B;
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " milliseconds\n";
	cout << "Time to add two 1000x1000 uniform random matrices\n";
    start = high_resolution_clock::now();
    C = A+B;
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " milliseconds\n";

}


int main () {
    NN_perf();
}