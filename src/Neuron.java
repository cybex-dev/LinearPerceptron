import java.io.File;
import java.io.FileNotFoundException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Neuron {

    public static class TestMark {
        public Double t1, t2, t3, exam;

        public TestMark(Double t1, Double t2, Double t3, Double exam) {
            this.t1 = t1;
            this.t2 = t2;
            this.t3 = t3;
            this.exam = exam;
        }

        List<Double> getMarks() {
            List<Double> l =new ArrayList<>();
            l.add(t1);
            l.add(t2);
            l.add(t3);
            return l;
        }

        public static List<TestMark> loadTestValues() {
            List<TestMark> testMarks = loadData("/Streamdata.csv");
            return testMarks.subList(testMarks.size() - 10, testMarks.size());
        }

        public static List<TestMark> loadTrainingValues() {
            List<TestMark> testMarks = loadData("/Streamdata.csv");
            return testMarks.subList(0, testMarks.size() - 10);
        }

        public static List<TestMark> loadEvaluationValues() {
            return loadData("/Evaluation.csv");
        }

        public static List<TestMark> loadData(String filename) {

            List<TestMark> l = new ArrayList<>();

            try {
                URI resource = Neuron.class.getResource(filename).toURI();

                File f = new File(resource);
                Scanner scanner = new Scanner(f);

                scanner.nextLine();
                while (scanner.hasNext()) {
                    String[] split = scanner.nextLine().split(",");
                    Double tt = 0.0;
                    tt = (split.length < 4) ? 0.0 : Double.parseDouble(split[3]);
                    TestMark t = new TestMark(Double.parseDouble(split[0]), Double.parseDouble(split[1]), Double.parseDouble(split[2]), tt);
                    l.add(t);
                }
            } catch (URISyntaxException | FileNotFoundException e) {
                e.printStackTrace();
            }

            return l;
        }

    }


    private List<Double> inputVector;
    private List<Double> inputWeights;
    private Double learningRate;
    private Double neuronOutput;
    private Double fNet;
    private Double biasWeight = 0.3; //can be random values
    private Double biasVector = 1.0;

    public Neuron(Double learningRate, Double initWeights) {
        this.learningRate = learningRate;
        this.biasWeight = initWeights;

        inputWeights = Arrays.asList(initWeights, initWeights, initWeights);
    }

    void updateInputVector(List<Double> inputVector) {
        this.inputVector = inputVector;
    }

    void calcFNet() {
        fNet = 0.0;
        for (int k = 0; k < inputWeights.size(); k++) {
            fNet += inputVector.get(k) * inputWeights.get(k);
        }
        fNet += biasVector * biasWeight;
    }

    Double calcNeuronOutput() {
        neuronOutput = fNet;
        return neuronOutput;
    }

    void updateInputWeights(Double neuronTargetOutput) {
        for (int k = 0; k < inputWeights.size(); k++) {
            double v = inputWeights.get(k) + learningRate * (neuronTargetOutput - neuronOutput) * 2.0 * inputVector.get(k);
            inputWeights.set(k, v);
        }
        biasWeight = biasWeight + learningRate * 2.0 * (neuronTargetOutput - neuronOutput) * biasVector;
    }

    public static void main(String[] args) {
        Neuron AN = new Neuron(0.0000001, 0.3);
        int count = Integer.MAX_VALUE; // number of iterations

        //training patterns
        List<TestMark> trainingDataInputs = TestMark.loadTrainingValues();

        //test patterns
        List<TestMark> testDataInputs = TestMark.loadTestValues();

        //evaluation patterns
        List<TestMark> evaluationInputs = TestMark.loadEvaluationValues();

        Double SSE = Double.MAX_VALUE, oldSSE = 0.0;

        //AN training phase
        int m = -1;
        Double p = 0.0;

        for (int i = 0; i < count; i++) {
            oldSSE = SSE;
            SSE = 0.0;

            for (m = 0; m < trainingDataInputs.size(); m++) {
                AN.updateInputVector(trainingDataInputs.get(m).getMarks());
                AN.calcFNet();
                SSE += Math.pow((trainingDataInputs.get(m).exam - AN.calcNeuronOutput()), 2);
                AN.updateInputWeights(trainingDataInputs.get(m).exam);
            }
            //training pattern set completed
//            if (i % 5000 == 0) {
//                System.out.printf("%f\n", SSE);
//            }

            // Test accuracy on testing set
            List<Double> percentErrors = new ArrayList<>();
            for (int n = 0; n < testDataInputs.size(); n++) {
                AN.updateInputVector(trainingDataInputs.get(n).getMarks());
                AN.calcFNet();
                Double exam = trainingDataInputs.get(n).exam;
                Double err = (exam - AN.fNet)/ exam;
                percentErrors.add(Math.abs(err));
            }

            p = percentErrors.stream().mapToDouble(value -> value).average().orElse(-1.0f);

            if (p < 0.10) {
                System.out.print("Training stopped since 10% error reached\n");
                System.out.printf("Error %f%%\n", p);
                break;
            }

            if (i % 50000 == 0) {
                System.out.printf("Error %f%%\n", p);
            }

            if (oldSSE <= SSE) {
                System.out.print("Training stopped since SSE(t-1) == SSE(t)\n");
                System.out.print("We do not want to overtrain\n");
                i = count;
                break;
            }
        }
        //completed number of iterations

//        System.out
        System.out.printf("Weights: \n1. %f\n2. %f\n3. %f\nBias %f\n", AN.inputWeights.get(0), AN.inputWeights.get(1), AN.inputWeights.get(2), AN.biasWeight);


        // Prediction
/*        for (int m = 0; m < evaluationInputs.size(); m++) {
            AN.updateInputVector(evaluationInputs.get(m).getMarks());
            AN.calcFNet();
            System.out.printf("Neuron Output = %f\n", AN.fNet);
        }*/
    }





}
