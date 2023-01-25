#include "./net.cpp"

void example1() {
	Mat<float> A(3,3), B(3,3), C(1,3), x(3,1);
	A.ind(0,0) = 1; A.ind(1,1) = 1; A.ind(2,2) = 1;
	B.ind(0,0) = 1; B.ind(1,1) = 1; B.ind(2,2) = 1;
	C.ind(0,0) = 1; C.ind(0,1) = 1; C.ind(0,2) = 1;
	x.ind(0,0) = 1; x.ind(1,0) = 0; x.ind(2,0) = 0;

    Mat<float> target(1,1), loss(1,1);
    Mat<float> Lx, LA, LB, LC;
    float lr=0.001;
    
    target.ind(0,0) = 5;
    Mat<float> y = nn.at(x,A,B,C);
    auto err = y -target;
    loss = square.at(err);
    cout << "Output of the NN is: " <<y.ind(0,0) << " the loss is: " << loss.ind(0,0) << endl;
    tie(Lx, LA, LB, LC) = nn.differential_at(x,A,B,C);
    LA.reshape(A); LB.reshape(B); LC.reshape(C);
    cout << "LA: \n";
    LA.print();
    cout << "LB: \n";
    LB.print();
    cout << "LC: \n";
    LC.print();
    cout << " ======= \n";

}

int main() {
	Mat<float> A(3,3), B(3,3), C(1,3), x(3,1);
	// A.random(); B.random(); C.random();
	A.ind(0,0) = 1; A.ind(1,1) = 1; A.ind(2,2) = 1;
	B.ind(0,0) = 1; B.ind(1,1) = 1; B.ind(2,2) = 1;
	C.ind(0,0) = 1; C.ind(0,1) = 1; C.ind(0,2) = 1;
	x.ind(0,0) = 1; x.ind(1,0) = 0; x.ind(2,0) = 0;
	// x.ind(0,0) = 1; x.ind(1,0) = -1; x.ind(2,0) = 2;

    Mat<float> target(1,1), loss(1,1);
    Mat<float> Lx, LA, LB, LC;
    float lr=0.01;
    
    target.ind(0,0) = 5;
    for (int t=0; t<1000; t++) {
	    Mat<float> y = nn.at(x,A,B,C);
        auto err = y -target;
        loss = square.at(err);
        cout << "Output of the NN is: " <<y.ind(0,0) << " the loss is: " << loss.ind(0,0) << endl;
        tie(Lx, LA, LB, LC) = nn.differential_at(x,A,B,C);
        Lx = square.differential_at(err) * Lx;
        LA = square.differential_at(err) * LA;
        LB = square.differential_at(err) * LB;
        LC = square.differential_at(err) * LC;
        LA.reshape(A); LB.reshape(B); LC.reshape(C);
        // cout << "LA: \n";
        // LA.print();
        // cout << "LB: \n";
        // LB.print();
        // cout << "LC: \n";
        // LC.print();
        // cout << " ======= \n";

        A = A-lr*LA;
        B = B-LB*lr;
        C = C-LC*lr;

    }
    return 0;
}