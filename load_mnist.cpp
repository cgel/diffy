#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <array>
using namespace std;

using image_type = array<int, 28*28>;

struct datapoint_type{
    image_type image;
    int label;
    void print() {
        cout << "Label: " << label << endl;
        cout << "Image: ";
        for (int k=0; k<28*28; k++) {cout << image[k] << ',';}
        cout << endl;
    }
};

// Couldn't find the original mnist files. Lecun's website requries username + password now
// The files in datasets use a different formatting

class MNIST {
    datapoint_type parse_line(string line) {
        stringstream ss(line);
        string num;
        datapoint_type datapoint;
        getline(ss, num, ',');
        // the first element contains the label
        datapoint.label = stoi(num);
        for (int i=0; i<28*28; i++) {
            getline(ss, num, ',');
            datapoint.image[i] = stoi(num);
        }
        return datapoint;
    }
    public:
    // Cannot allocate the entire array on the stack
    datapoint_type* train_datapoints;
    datapoint_type* test_datapoints;
    int train_size;
    int test_size;

    MNIST(int tr_size=60000, int te_size=10000) {
        assert( tr_size>0 and tr_size<=60000);
        assert( te_size>0 and te_size<=10000);
        train_size= tr_size;
        test_size= te_size;

        train_datapoints = new datapoint_type[train_size];
        test_datapoints = new datapoint_type[test_size];
        string line;
        ifstream trfile;
        trfile.open("./datasets/mnist_train.csv");
        if (trfile.is_open()) {
            for (int i=0; i<train_size; i++) {
                getline(trfile, line);
                train_datapoints[i] = parse_line(line); 
            }
        }
        else {
            cout << "Could not read mnist_train.csv\n";
        }
        ifstream tefile;
        tefile.open("./datasets/mnist_test.csv");
        if (tefile.is_open()) {
            for (int i=0; i<test_size; i++) {
                getline(tefile, line);
                test_datapoints[i] = parse_line(line); 
            }
        }
        else {
            cout << "Could not read mnist_test.csv\n";
        }
 
    }
    ~MNIST() {
        delete train_datapoints;
        delete test_datapoints;
    }
};