//multilevel neural net framework
//you need to fill in the getRawPrediction and train functions

public class Perceptron
{
    private static final double ALPHA=0.05;
    private static final double NOISEMAX=0.4;

    //weights from hidden to output layers
    double[] outputweight;
    //weights from input to hidden layers
    double[][] hiddenweight;

    //temporary space for caching hidden layer values
    double[] hidden;

    //number of nodes in input and hidden layers (size of neural network)
    int size;

    //constructor.  Called with the number of inputs:  new Perceptron(3) makes a three input, two output perceptron.
    Perceptron(int size)
    {
        // how many nodes we have
        this.size=size;
        //make an array of weights from each hidden, plus a bias, to each output node (size+1) = accounting for bias
        outputweight=new double[size+1];
        //make a 2D array of weights from each input, plus a bias, to each hidden node
        hiddenweight=new double[size][size+1];
        for(int i=0; i<size+1; i++)
            outputweight[i]=Math.random()*NOISEMAX-NOISEMAX/2;
        for(int i=0; i<size; i++)
            for(int j=0; j<size+1; j++)
                // gives random num between -0.2 to 0.2. Give it a nudge
                hiddenweight[i][j]=Math.random()*NOISEMAX-NOISEMAX/2;
        //create the array for caching, but don't bother initializing it
        hidden=new double[size];
    }

    //returns whether the raw prediction is a 1 or 0
    int getPrediction(int[] inputs)
    {
        return getRawPrediction(inputs)>=0.5? 1:0;
    }

    //takes an array of inputs in range 0 to 1, feeds them to the perceptron, saves a guess in range 0 to 1 in array "outputs"
    double getRawPrediction(int[] inputs)
    {
        //TODO: DONE?

        // for each hidden node, multiply inputs by weights, sum, threshold
        for (int h = 0; h < size; h++) {        // or hidden.length?

            double total = 0;

            //1. rescale the inputs from -1 to 1 and copy them to array inputs
            for (int i = 0; i < size; i++) {        // or inputs.length?
                int neuralInput = inputs[i];
                if (neuralInput == 0) {
                    neuralInput = -1;
                } else {
                    neuralInput = 1;
                }
                //2. compute dot product of inputs times weights for each hidden... (total for this particular weight [h][i])
                total += hiddenweight[h][i] * neuralInput;
            }
            // adding the bias
            total += hiddenweight[h][hiddenweight.length-1];

            // Applying threshold and saving it (do sigmoid of total and save it in array hidden)
            hidden[h] = sigmoid(total);
        }

        //3. compute dot product of hidden times weights for each output.  do sigmoid and return it
        double total = 0;
        for (int h = 0; h < size; h++) {
            total += outputweight[h] * hidden[h];
        }
       // Add on bias weight
        total += outputweight[outputweight.length-1];
        double output = sigmoid(total);
        return output;

    }

    //this trains the perceptron on an array of inputs (1/0) and desired outputs (1/0)
//the weights are adjusted and errors are saved in array "error".  return TRUE if training is done
    boolean train(int[] inputs, int want)
    {
        //TODO:
        //1. call getRawPrediction on inputs.  this will put values in hidden and outputs that we can use for training
        double prediction = getRawPrediction(inputs);

        //2. compute output error for each output and save it in "errors":  error = desired-predicted (loss)
        double error = prediction - want;

        //3. compute output training error for each output node:  outTrainError = error * predicted * (1-predicted)
        double outTrainError = error * prediction * (1-prediction);

        //4. compute hidden error for each hidden node:  hiddenError = sum of (outTrainError * output weight) over all outputs  SUM?
        double[] hiddenError = new double[size];
        double hiddenTrainError = 0;
        // getting my array of hidden errors
        for (int h = 0; h < size; h++) {
            hiddenError[h] = hidden[h] * (1-hidden[h]) * outputweight[h] * outTrainError;
        }

        //5. for each hidden node, apply output training error to weights:  outputweight += alpha * outTrainError * hidden-value
        for (int h = 0; h < size; h++) {
            outputweight[h] += outTrainError * hidden[h] * ALPHA;
        }
        //don't forget to train the bias weight.  it has a hidden-value of 1
        outputweight[outputweight.length-1] += outTrainError * ALPHA;

        int theInput;
        //6. over each input, compute hidden training error: hiddenTrainError = hiddenError * hidden-value * (1-hidden-value)
        for (int h = 0; h < size; h++) {
            for (int i = 0; i < size; i++) {

                hiddenTrainError = hiddenError[h] * hidden[h] * (1-hidden[h]);

                //7. apply that error to the input weight: hiddenweight += hiddentrainingerror * inputvalue * (1-inputvalue)
                hiddenweight[h][i] += hiddenTrainError * inputs[h] * (1-inputs[h]);
            }
        }


        // LOOK THIS ONE OVER, EVERYTHING ELSE SEEMS RIGHT UP TOP (1-6)
        // PUT IN OCR??   ----V

        double maxError = 0;
        //8. go through all the errors in the array and keep track of the maximum.
        for (int h = 0; h < size; h++) {
            if (hiddenError[h] > maxError) {
                maxError = hiddenError[h];
            }
        }
        // if the max error is below some threshold (say 0.1), return TRUE. (else FALSE)
        if (maxError > 0.9 || maxError < 0.1) {
            return true;
        } else {
            return false;	//replace this line
        }
    }

    //implements the threshold function 1/(1+e^-x)
//this is mathematically close to the >=0 threshold we use in the single layer perceptron, but is differentiable
    static double sigmoid(double x)
    {
        return 1.0/(1.0+Math.pow(2.71828,-x));
    }
}