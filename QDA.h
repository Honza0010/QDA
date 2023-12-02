#ifndef __QDA_
#define __QDA_

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <armadillo>

class QDA
{
	int k;	//Number of classes
	int m;	//Dimension
	std::vector<arma::mat> data;
	std::vector<arma::mat> means;
	std::vector<arma::mat> cov_matrices;

public:
	QDA(int k, int m, const std::string& filename);

	int predict_class(std::vector<double> input_data);

private:
	void load_file(const std::string& filename);

	void calculate_means();

	void calculate_cov_matrices();

	void write()
	{
		std::cout << data[0] << std::endl;

		for (int i = 0; i < k; i++)
		{
			//std::cout << means[i] << std::endl;
			std::cout << cov_matrices[i] << std::endl;
		}
	}
};


QDA::QDA(int k, int m, const std::string& filename)
	: k(k), m(m)
{
	data = std::vector<arma::mat>(k, arma::mat(0, m));
	means = std::vector<arma::mat>(k, arma::mat(1, m));
	cov_matrices = std::vector<arma::mat>(k, arma::mat());

	load_file(filename);

	calculate_means();

	calculate_cov_matrices();

	write();
}



void QDA::load_file(const std::string& filename)
{
	std::ifstream file;
	file.open(filename);  //otevru soubor pro zapis
	if (file.is_open())
	{
		int x;
		while (!file.eof())
		{
			
			std::vector<double> help(m + 1, 0);
			for (int i = 0; i <= this->m; i++)
			{
				if (file.eof())
				{
					throw std::logic_error("Bad format of dataset");
				}
				file >> help[i];
				if (file.fail())
				{
					throw std::logic_error("Bad values in dataset");
				}
			}

			int index = help[m]-1;

			arma::mat& class_matrix = data[index];
			class_matrix.resize(class_matrix.n_rows + 1, m);
			for (int i = 0; i < m; i++)
			{
				class_matrix(class_matrix.n_rows-1, i) = help[i];
			}
			//data[index].resize(data[index].n_rows + 1, m);
		}
		file.close();
	}
	else
	{
		throw std::invalid_argument("File could not have been opened");
	}
}

void QDA::calculate_means()
{
	for (int i = 0; i < this->k; i++)
	{
		means[i] = arma::mean(data[i]);
	}
}

void QDA::calculate_cov_matrices()
{
	for (int i = 0; i < this->k; i++)
	{
		cov_matrices[i] = arma::cov(data[i]);
	}
}


inline int QDA::predict_class(std::vector<double> input_data)
{
	if (input_data.size() != this->m)
	{
		throw std::logic_error("Bad dimension of input data");
	}

	arma::mat help(1,m);
	for (int i = 0; i < this->m; i++)
	{
		help(0, i) = input_data[i];
	}

	std::cout << help << std::endl;

	return 0;
}

#endif // !__QDA_
