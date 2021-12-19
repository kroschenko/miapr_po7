using System;
using System.Collections.Generic;
using System.IO;
using Miapr;


namespace Miapr1
{
    sealed class NeuralNetworkWHoff : NeuralNetwork, IErrors
    {
        public List<double> Errors { get; } = new List<double>();

        public NeuralNetworkWHoff(int _inputsNum) : base(_inputsNum) { }

        public override double Activate(params double[] inputs)
        {
            double sum = 0;
            for (int i = 0; i < inputsNum; i++)
            {
                sum += weights[i] * inputs[i];
            }

            return sum - threshold;
        }

        public override void Training(double[] trainingSample)
        {
            double changeSpeed = 0.1;
            double minError = 0.01;
            double currError = 0;
            double[] slice = new double[inputsNum];
            int epochs = 1;

            Console.WriteLine("NN result   Real result   Error");
            do
            {
                currError = 0;
                for (int i = 0; i < trainingSample.Length - inputsNum; i++)
                {
                    changeSpeed = ChangeSpeed(trainingSample, inputsNum, i);
                    Array.Copy(trainingSample, i, slice, 0, inputsNum);
                    double output = Activate(slice);
                    Console.WriteLine($"{output:f7}  {trainingSample[i + inputsNum]:f9}  {output - trainingSample[i + inputsNum]}");
                    for (int j = 0; j < inputsNum; j++)
                    {
                        weights[j] -= changeSpeed * (output - trainingSample[i + inputsNum]) * slice[j];
                    }
                    threshold += changeSpeed * (output - trainingSample[i + inputsNum]);
                    currError += 0.5 * Math.Pow((output - trainingSample[i + inputsNum]), 2); ;
                    Errors.Add(currError);
                }
                epochs++;
            } while (currError >= minError);
            Console.WriteLine($"Epohs: {epochs}");
        }

        public void WriteErrorsToFile(string path)
        {
            StreamWriter sw = new StreamWriter(path, false);
            foreach (var error in Errors)
            {
                sw.Write(error);
                sw.Write(" ");
            }
            sw.Close();
        }
       
        private double ChangeSpeed(double[] trainingSample, int inputsNum, int i)
        {
            double speedTraining = 0;
            for (int j = 0; j < inputsNum; j++)
            {
                speedTraining += Math.Pow(trainingSample[i + j], 2);
            }
            return 1 / (1 + speedTraining);
        }
    }
}
