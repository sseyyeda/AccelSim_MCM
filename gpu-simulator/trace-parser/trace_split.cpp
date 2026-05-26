#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

using namespace std;

int main( int argc, char ** argv )
{
    if ( argc < 3 )
    {
        cerr << "Error: enter trace_file and chiplet_num.\n";
        exit(1);
    }
    string trace_file = argv[1];
    uint chiplet_num = stoi(argv[2]);
    if (chiplet_num > 256)
    {
        cerr << "Error: only support chiplet_num < 256.\n";
        exit(1);
    }

    ifstream ifs;
    ifs.open(trace_file.c_str());
    if (!ifs.is_open()) {
    	std::cout << "Unable to open file: " << trace_file << std::endl;
    	exit(1);
	  }

	  uint gridDim_x, gridDim_y, gridDim_z;
	  uint total_cta;
	  uint cta_per_chiplet[256]; // max chiplet number 256 for now

    string line;
    int chiplet_id = 0;
    int end_cta, cta_x, cta_y, cta_z;
    bool finish_chiplet_trace = false;
    bool last_cta = false;
    string filename = trace_file + "_" + to_string(chiplet_id);
    cout << "Writing to file: " << filename << std::endl;
    ofstream ofs[256];
    ofs[chiplet_id].open(filename.c_str());


    while (!ifs.eof()) {
    	getline(ifs, line);
        ofs[chiplet_id] << line << "\n";
    	if (line.length() == 0) {
      		continue;
    	} else {
      		std::stringstream ss;
      		std::string string1, string2;
      		ss.str(line);
      		//ss.ignore();
      		ss >> string1 >> string2;

      		if (string1 == "-grid" && string2 == "dim") {
      			sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &gridDim_x, &gridDim_y, &gridDim_z);
      			total_cta = gridDim_x * gridDim_y * gridDim_z;
      			cout << "Total CTA number = " << total_cta << ". Grid_Dim(" << gridDim_x << ", " << gridDim_y << ", " << gridDim_z << ")" << std::endl;
      			for (int i = 0; i < chiplet_num; i++) {
      				cta_per_chiplet[i] = total_cta / chiplet_num;
    				if (i < total_cta % chiplet_num)
    					cta_per_chiplet[i] ++;
    				cout << "Chiplet " << i << " has " << cta_per_chiplet[i] << " CTAs." << std::endl;
      			}
      			end_cta = cta_per_chiplet[0] - 1;

      		} else if (string1 == "thread" && string2 == "block") {
      			sscanf(line.c_str(), "thread block = %d,%d,%d", &cta_x, &cta_y, &cta_z);
      			if ((cta_x + cta_y * gridDim_x + cta_z * gridDim_x * gridDim_y) == end_cta) {
      				if (chiplet_id == (chiplet_num - 1)) 
      					last_cta = true;
      				finish_chiplet_trace = true;
      			}

      		} else if (string1 == "#END_TB") {
      			if ( finish_chiplet_trace ) {
      				ofs[chiplet_id].close();
      				if (last_cta) break;
      				chiplet_id ++;
      				end_cta += cta_per_chiplet[chiplet_id];
      				filename = trace_file + "_" + to_string(chiplet_id);
      				cout << "Writing to file: " << filename << std::endl;
      				ofs[chiplet_id].open(filename.c_str());
      				finish_chiplet_trace = false;
      			}
      		}
      	}
    }

    if (ifs.is_open()) ifs.close();

}
