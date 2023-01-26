#include "./mat.cpp"
#include <string.h>
#include <chrono>
using namespace std::chrono;

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

int main () {
	cout << "Test correctness of matmul \n";
	Mat<float> m(2,3);
	m.ind(0,0) = 1; m.ind(0,1) = 0; m.ind(0,2) = 1;
	m.ind(1,0) = 0; m.ind(1,1) = 1; m.ind(1,2) = 1;

	Mat<float> n(3,3);
	n.ind(0,0) = 0; n.ind(0,1) = 0; n.ind(0,2) = 0;
	n.ind(1,0) = 1; n.ind(1,1) = 1; n.ind(1,2) = 1;
	n.ind(2,0) = 0; n.ind(2,1) = 1; n.ind(2,2) = 1;

    Mat<float> x = m*n;

    Mat<float> y(2,3);
	y.ind(0,0) = 0; y.ind(0,1) = 1; y.ind(0,2) = 1;
	y.ind(1,0) = 1; y.ind(1,1) = 2; y.ind(1,2) = 2;
    cout << "x==y " << BoolToString(y == x) << '\n';

	cout << "Creating 5x5 random uniform matrix \n";
    Mat<float> s(5,5);
    s.random(0, 1);
    s.print();

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