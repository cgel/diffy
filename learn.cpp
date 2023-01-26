#include "./net.cpp"
#include "./load_mnist.cpp"
#include <cmath>

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

Mat<float> image_to_mat(array<int, 28*28>& image) {
    Mat<float> mat(28*28, 1);
    for (int i=0; i<28*28; i++ ) {
        mat.ind(i,0) = ((float)image[i])/256.;
    }
    return mat;
}

void mnist_train() {
    MNIST mnist(10,10);
    int hidden_dim = 50;
    int input_dim = 28*28;

	Mat<float> A(hidden_dim, input_dim), B(hidden_dim,hidden_dim), C(1,hidden_dim);
    // A.random(0, sqrt(2/input_dim)); B.random(0, sqrt(2/hidden_dim)); C.random(0, sqrt(2/hidden_dim));
    A.random(0, (2./input_dim)); B.random(0, (2./hidden_dim)); C.random(0, (2./hidden_dim));
    
    Mat<float> Lx, LA, LB, LC, loss;
    float lr=0.001;
    Mat<float> err;
    for (int t=0; t<1000; t++) {
        datapoint_type datapoint = mnist.train_datapoints[t%mnist.train_size];
        Mat<float> img = image_to_mat(datapoint.image);
	    Mat<float> y = nn.at(img,A,B,C);
        err = y-datapoint.label;
        loss = square.at(err);
        cout << "Step: " << t<< " NN output: " <<y.ind(0,0) << " Target: " << datapoint.label << " Loss: " << loss.ind(0,0) << endl;
        tie(Lx, LA, LB, LC) = nn.differential_at(img,A,B,C);
        LA = square.differential_at(err) * LA;
        LB = square.differential_at(err) * LB;
        LC = square.differential_at(err) * LC;
        LA.reshape(A); LB.reshape(B); LC.reshape(C);
        A = A-lr*LA;
        B = B-LB*lr;
        C = C-LC*lr;
        if (y.ind(0,0) == 0.) {
            cout << "Output looking weird. The norms of the matrices:\n";
            cout << A.norm() << " "  << B.norm() << " "  << C.norm() << endl;
            cout << "The norms of the differentials:\n";
            cout << LA.norm() << " "  << LB.norm() << " "  << LC.norm() << endl;
        }
        cout << "---------\n";
    }
}

int main() {
    mnist_train();
    return 0;
}