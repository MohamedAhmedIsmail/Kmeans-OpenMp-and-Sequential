#include<iostream>
#include<string>
#include<fstream>
#include<vector>
#include <sstream>
#include<math.h>
#include <omp.h>
using namespace std;
class ReadMyFile
{
public:
	vector<vector<float>> ReadMyData(string Path)
	{
		fstream file(Path);
		string line;
		vector<vector<float>>myVector;
		int i = 0;

		while (getline(file, line))
		{
			float value;
			stringstream ss(line);
			myVector.push_back(vector<float>());
			while (ss >> value)
			{
				myVector[i].push_back(value);
			}
			i++;
		}
		return myVector;
	}
};
class K_mean_Sequential
{
private:
	vector<vector<float>>myVector;
public:
	K_mean_Sequential(vector<vector<float>>Vector)
	{
		myVector = Vector;
	}
	void K_means_clusters()
	{
		vector<vector<float>>kVectors;
		int x = 0;
		for (int i = 0; i < myVector.size(); i++)
		{
			kVectors.push_back(vector<float>());
			for (int j = 0; j < myVector[i].size(); j++)
			{
				kVectors[x].push_back(myVector[i][j]);
			}
			x++;
		}
		vector<vector<float>>mean;
		int k = 5;
		float distance[50][150] = { 0 };
		int counterArr[1002] = { 0 };
		vector<vector<float>>meantemp;
		for (int i = 0; i < k; i++)
		{
			mean.push_back(vector<float>());
			mean[i].push_back(kVectors[i][0]);
			mean[i].push_back(kVectors[i][1]);
			mean[i].push_back(kVectors[i][2]);
			mean[i].push_back(kVectors[i][3]);
			meantemp.push_back(vector<float>());
			for (int j = 0; j < 4; j++)
			{
				meantemp[i].push_back(0.0);
			}
		}
		vector<vector<vector<float>>>outClusters;
		while (true)
		{
			float minimum = 100000;
			int clnum = -1;
			float sum[160][160] = { 0 };
			int count = 0;
			for (int i = 0; i < k; i++)
			{
				for (int j = 0; j < kVectors.size(); j++)
				{
					distance[i][j] = sqrt(pow((mean[i][0] - kVectors[j][0]), 2) +pow((mean[i][1] - kVectors[j][1]), 2) +pow((mean[i][2] - kVectors[j][2]), 2) +pow((mean[i][3] - kVectors[j][3]), 2));
				}
			}
		outClusters.clear();
		for (int j = 0; j < k; j++)
			outClusters.push_back(vector<vector<float>>());
			for (int j = 0; j < kVectors.size(); j++)
			{
				for (int i = 0; i < k; i++)
				{
					if (distance[i][j] < minimum)
					{
						minimum = distance[i][j];
						clnum = i;
					}
				}
				sum[clnum][0] += kVectors[j][0];
				sum[clnum][1] += kVectors[j][1];
				sum[clnum][2] += kVectors[j][2];
				sum[clnum][3] += kVectors[j][3];
				counterArr[clnum] += 1;
				outClusters[clnum].push_back(kVectors[j]);
				minimum = 10000000;
				clnum = -1;
			}
			for (int i = 0; i < k; i++)
			{
				
				meantemp[i][0] = sum[i][0] / (1.0*counterArr[i]);
				meantemp[i][1] = sum[i][1] / (1.0*counterArr[i]);
				meantemp[i][2] = sum[i][2] / (1.0*counterArr[i]);
				meantemp[i][3] = sum[i][3] / (1.0*counterArr[i]);
				counterArr[i] = 0;
			}
		   for (int i = 0; i < k; i++)
		   {
			   if (int(mean[i][0]) == int(meantemp[i][0])&& int(mean[i][1]) == int(meantemp[i][1])&& int(mean[i][2]) == int(meantemp[i][2]) && int(mean[i][3]) == int(meantemp[i][3]))
			   {   
				   count++;
			   }
		   }
		   if (count == k)
			   break;
		   else
		   {
			   for (int i = 0; i < k; i++)
			   {
				   mean[i][0] = meantemp[i][0];
				   mean[i][1] = meantemp[i][1];
				   mean[i][2] = meantemp[i][2];
				   mean[i][3] = meantemp[i][3];
			   }
		   }
		}
		/*
		for (int i = 0; i < k; i++)
		{
			cout << "Cluster " << i + 1 << endl;
			for (int j = 0; j < outClusters[i].size(); j++)
			{
				cout << outClusters[i][j][0] << " " << outClusters[i][j][1] << " " << outClusters[i][j][2] <<" "<<outClusters[i][j][3]<< endl;
			}
			cout << endl;
		}*/
		float sum1[5] = { 0 }, sum2[5] = { 0 }, sum3[5] = { 0 }, sum4[5] = { 0 };
		for (int i = 0; i < k; i++)
		{
			//cout << outClusters[i].size() << endl;
			for (int j = 0; j < outClusters[i].size(); j++)
			{
				sum1[i] += outClusters[i][j][0];
				sum2[i] += outClusters[i][j][1];
				sum3[i] += outClusters[i][j][2];
				sum4[i] += outClusters[i][j][3];
			}
		}
		for (int i = 0; i < k; i++)
		{
			cout <<i<<" "<< sum1[i]/outClusters[i].size() <<" "<<sum2[i]/outClusters[i].size()<<" "<<sum3[i]/outClusters[i].size()<<" "<<sum4[i]/outClusters[i].size()<< endl;
		}
	}
};
class K_means_Parallel
{
private:
	vector<vector<float>>myVector;
public:
	K_means_Parallel(vector<vector<float>>Vector)
	{
		myVector = Vector;
	}
	void K_means_clusters()
	{
		omp_set_num_threads(4);
		int id;
		vector<vector<float>>kVectors;
		#pragma omp parallel private(id)
		{
			id = omp_get_thread_num();
			vector<vector<float>>kVectors;
			int x = 0;
			#pragma omp for schedule(static,4)
			for (int i = 0; i < myVector.size(); i++)
			{
				kVectors.push_back(vector<float>());
				for (int j = 0; j < myVector[i].size(); j++)
				{
					kVectors[x].push_back(myVector[i][j]);
				}
				x++;
			}
			vector<vector<float>>mean;
			int k = 5;
			float distance[50][150] = { 0 };
			int counterArr[1002] = { 0 };
			vector<vector<float>>meantemp;
			for (int i = 0; i < k; i++)
			{
				mean.push_back(vector<float>());
				mean[i].push_back(kVectors[i][0]);
				mean[i].push_back(kVectors[i][1]);
				mean[i].push_back(kVectors[i][2]);
				mean[i].push_back(kVectors[i][3]);
				meantemp.push_back(vector<float>());
				for (int j = 0; j < 4; j++)
				{
					meantemp[i].push_back(0.0);
				}
			}
			vector<vector<vector<float>>>outClusters;
			while (true)
			{
				float minimum = 100000;
				int clnum = -1;
				float sum[160][160] = { 0 };
				int count = 0;
				for (int i = 0; i < k; i++)
				{
					for (int j = 0; j < kVectors.size(); j++)
					{
						distance[i][j] = sqrt(pow((mean[i][0] - kVectors[j][0]), 2) + pow((mean[i][1] - kVectors[j][1]), 2) + pow((mean[i][2] - kVectors[j][2]), 2) + pow((mean[i][3] - kVectors[j][3]), 2));
					}
				}
				outClusters.clear();
				for (int j = 0; j < k; j++)
					outClusters.push_back(vector<vector<float>>());
				for (int j = 0; j < kVectors.size(); j++)
				{
					for (int i = 0; i < k; i++)
					{
						if (distance[i][j] < minimum)
						{
							minimum = distance[i][j];
							clnum = i;
						}
					}
					sum[clnum][0] += kVectors[j][0];
					sum[clnum][1] += kVectors[j][1];
					sum[clnum][2] += kVectors[j][2];
					sum[clnum][3] += kVectors[j][3];
					counterArr[clnum] += 1;
					outClusters[clnum].push_back(kVectors[j]);
					minimum = 10000000;
					clnum = -1;
				}
				for (int i = 0; i < k; i++)
				{

					meantemp[i][0] = sum[i][0] / (1.0*counterArr[i]);
					meantemp[i][1] = sum[i][1] / (1.0*counterArr[i]);
					meantemp[i][2] = sum[i][2] / (1.0*counterArr[i]);
					meantemp[i][3] = sum[i][3] / (1.0*counterArr[i]);
					counterArr[i] = 0;
				}
				for (int i = 0; i < k; i++)
				{
					if (int(mean[i][0]) == int(meantemp[i][0]) && int(mean[i][1]) == int(meantemp[i][1]) && int(mean[i][2]) == int(meantemp[i][2]) && int(mean[i][3]) == int(meantemp[i][3]))
					{
						count++;
					}
				}
				if (count == k)
					break;
				else
				{
					for (int i = 0; i < k; i++)
					{
						mean[i][0] = meantemp[i][0];
						mean[i][1] = meantemp[i][1];
						mean[i][2] = meantemp[i][2];
						mean[i][3] = meantemp[i][3];
					}
				}
			}
			float sum1[5] = { 0 }, sum2[5] = { 0 }, sum3[5] = { 0 }, sum4[5] = { 0 };
#pragma omp sections
			{
#pragma omp section
				{
					for (int i = 0; i < k; i++)
					{
						for (int j = 0; j < outClusters[i].size(); j++)
						{
							sum1[i] += outClusters[i][j][0];
						}
					}
				}
#pragma omp section
				{
				for (int i = 0; i < k; i++)
				{
					for (int j = 0; j < outClusters[i].size(); j++)
					{
						sum2[i] += outClusters[i][j][1];
					}
				}
			}
#pragma omp section
				{
					for (int i = 0; i < k; i++)
					{
						for (int j = 0; j < outClusters[i].size(); j++)
						{
							sum3[i] += outClusters[i][j][2];
						}
					}
				}
#pragma omp section
				{
					for (int i = 0; i < k; i++)
					{
						for (int j = 0; j < outClusters[i].size(); j++)
						{
							sum4[i] += outClusters[i][j][3];
						}
					}
				}
			}
			for (int i = 0; i < k; i++)
			{
				//printf("The Thread Number %d , and the number %d \n", id, i);
				printf(" %d\n %f \n %f \n %f \n %f \n %f\n", i,sum1[i] / outClusters[i].size(), sum2[i] / outClusters[i].size(), sum3[i] / outClusters[i].size(), sum4[i] / outClusters[i].size());
			}
		}
	}
};
int main()
{
	ReadMyFile myFile;
	vector<vector<float>>myVector;
	myVector=myFile.ReadMyData("C:\\Users\\mohamed ismail\\Desktop\\IrisDataset.txt");
	K_mean_Sequential k_meanSeq(myVector);
	cout << "The Coordinates of k-means in Sequentail:" << endl;
	k_meanSeq.K_means_clusters();
	K_means_Parallel k_meanParallel(myVector);
	cout << "The Coordinates of k-means in Parallel:" << endl;
	k_meanParallel.K_means_clusters();
	system("pause");
	return 0;
}
