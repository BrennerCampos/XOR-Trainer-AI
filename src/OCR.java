/*A character recognizer that uses neural nets

TODO: BRENNER CAMPOS

Michael Black, 11/2020

TODO:  
        YOUR CODE WILL GO IN FUNCTIONS test() AND train()
        HERE STATE WHAT STEPS YOU ACCOMPLISHED

usage:
ocr sample X
        pops up a window, user draws an example of an X, user doubleclicks and the X is saved for later
ocr train
        builds a neural net for each letter type, trains each of them on the samples until they predict perfectly
ocr test
        pops up a window, user draws a letter and doubleclicks, the program tries to guess which letter was drawn
*/

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;

public class OCR extends JComponent implements MouseListener, MouseMotionListener
{
    //global constants

    //squares wide
    public static final int GRIDWIDTH=10;
    //squared tall
    public static final int GRIDHEIGHT=20;
    //window dimensions
    public static final int SCREENWIDTH=400, SCREENHEIGHT=400;
    //flags
    public static final int SAMPLE=1,TRAIN=2,TEST=3;

    //array of grid squares: true=filled, false=empty
    public boolean[][] square;

    //operation being performed: SAMPLE, TRAIN, TEST
    public int operation;
    //for sample only, letter being drawn
    public char letter;

    private JFrame window;


    //read the contents of the grid and save them to the end of ocrdata.txt
    public void saveSample()
    {
        try
        {
            PrintWriter datafile=new PrintWriter(new FileOutputStream(new File("ocrdata.txt"),true));
            datafile.print(letter+" ");
            int[] data=getSquareData();
            for(int x=0; x<data.length; x++)
                datafile.print(data[x]);
            datafile.println();
            datafile.close();
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    }

    //called immediately on "ocr train"
//reads the images in ocrdata.txt, builds a set of neural nets, trains them, and saves the weights to perceptron.txt
    public void train()
    {
        try
        {
            Scanner ocrdata=new Scanner(new FileReader("ocrdata.txt"));
            int linecount=0;	//keep track of how many samples are in the file
            int sample_types=0;	//keep track of how many different types of letters are in the file
            //go through the file and just count the samples
            while(ocrdata.hasNextLine())
            {
                linecount++;
                ocrdata.nextLine();
            }
            ocrdata.close();

            //make an array to hold the samples
            int[][] sample_input=new int[linecount][GRIDWIDTH*GRIDHEIGHT];
            //make another array to hold the output letter for each sample
            char[] sample_output=new char[linecount];
            //reopen the file
            ocrdata=new Scanner(new FileReader("ocrdata.txt"));
            //for each sample,
            for(int i=0; i<linecount; i++)
            {
                String line=ocrdata.nextLine();
                //the first character is the output letter
                sample_output[i]=line.charAt(0);
                //then a space, then a 1 or 0 for each square
                for(int j=0; j<GRIDWIDTH*GRIDHEIGHT; j++)
                    sample_input[i][j] = (line.charAt(j+2)=='1'?1:0);

            }
            ocrdata.close();

//TODO: MAKE NEURAL NET (PERCEPTRON) OBJECTS AND TRAIN THEM HERE, THEN SAVE THE WEIGHTS TO perceptron.txt
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

        // end of train, put in XOR test code


        // 1. make a couple arrays to hold inputs and desiredOutputs:
        int[] in = new int[2];
        int[] out = new int[2];

        // 2. make a Perceptron object with two inputs and outputs
        Perceptron neuron_1 = new Perceptron(2);
        Perceptron neuron_0 = new Perceptron(2);

        int CUTOFF = 10000;
        int correctCount = 0;




        for (int itr = 0; itr<CUTOFF; itr++) {
            correctCount = 0;
           // boolean done = true;

            // 3.  train on the first row
            in[0]=0; in[1]=0; out[0]=0; out[1]=0;
            //System.out.println(neuron_1.getPrediction(in));
            boolean n1right = neuron_1.train(in, out[1]);
            boolean n0right = neuron_0.train(in, out[0]);
            if (n1right) correctCount++;
            if (n0right) correctCount++;


            // 4. train on the other rows
            in[0]=1; in[1]=0; out[0]=1; out[1]=0;
          //  System.out.println(neuron_1.getPrediction(in));
            n1right = neuron_1.train(in, out[1]);
            n0right = neuron_0.train(in, out[0]);
            if (n1right) correctCount++;
            if (n0right) correctCount++;

         //   if (!n1right || !n0right) done = false;
            in[0]=0; in[1]=1; out[0]=1; out[1]=0;
          //  System.out.println(neuron_1.getPrediction(in));
            n1right = neuron_1.train(in, out[1]);
            n0right = neuron_0.train(in, out[0]);
            if (n1right) correctCount++;
            if (n0right) correctCount++;

            in[0]=1; in[1]=1; out[0]=0; out[1]=1;
          //  System.out.println(neuron_1.getPrediction(in));
            n1right = neuron_1.train(in, out[1]);
            n0right = neuron_0.train(in, out[0]);
            if (n1right) correctCount++;
            if (n0right) correctCount++;

            if (correctCount == 8) {
                break;
            }
        }

        if (correctCount == 8) {
            System.out.println("learned it");
        }  else {
            System.out.println(" never earned it");
        }



        // test it: write four times
        in[0]=0; in[1]=0;
        System.out.println("The prediction for "+in[1]+" "+in[0]+": "+ neuron_1.getPrediction(in)+ " "+neuron_0.getPrediction(in));




        // 5. repeat 3 and 4 until every train returns correct

    }

    //called on "ocr test", after the user draws and right-clicks the mouse
    public void test()
    {
        //TODO: MAKE A NEURAL NET OBJECT, READ THE WEIGHTS FROM A FILE perceptron.txt, USE THE NEURAL NET TO IDENTIFY THE LETTER



    }

    //returns contents of all squares as array of 1 (filled) 0 (unfilled)
    public int[] getSquareData()
    {
        int[] data=new int[GRIDWIDTH*GRIDHEIGHT];
        for(int x=0; x<GRIDWIDTH; x++)
            for(int y=0; y<GRIDHEIGHT; y++)
                data[GRIDWIDTH*y+x]=square[x][y]? 1:0;
        return data;
    }

    public OCR(int operation)
    {
        this.operation=operation;
        if(operation==SAMPLE || operation==TEST)
            constructWindow();
        else if (operation==TRAIN)
            train();
    }
    public OCR(int operation, char letter)
    {
        this.operation=operation;
        this.letter=letter;
        if(operation==SAMPLE || operation==TEST)
            constructWindow();
    }


    public void drawingCompleted()
    {
        if(window!=null)
            window.setVisible(false);
        if(operation==SAMPLE)
            saveSample();
        else if (operation==TEST)
            test();
        System.exit(0);
    }

    public void constructWindow()
    {
        square=new boolean[GRIDWIDTH][GRIDHEIGHT];

        window=new JFrame("OCR");
        window.setSize(SCREENWIDTH+10,SCREENHEIGHT+30);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.add(this);
        addMouseListener(this);
        addMouseMotionListener(this);
        if(operation==SAMPLE)
            window.setTitle("OCR - draw letter "+letter+" and right-click when done");
        else if (operation==TEST)
            window.setTitle("OCR - draw a letter and right-click when done");
        window.setVisible(true);
    }

    public void paintComponent(Graphics g)
    {
        int squarewidth=SCREENWIDTH / GRIDWIDTH;
        int squareheight=SCREENHEIGHT / GRIDHEIGHT;
        for(int x=0; x<GRIDWIDTH; x++)
        {
            for(int y=0; y<GRIDHEIGHT; y++)
            {
                if(square[x][y])
                    g.setColor(new Color(100,100,0));
                else
                    g.setColor(new Color(255,255,255));
                g.fillRect(x*squarewidth,y*squareheight,squarewidth,squareheight);
            }
        }
    }


    public void mousePressed(MouseEvent e)
    {
        if(e.getButton()==MouseEvent.BUTTON1)
        {
            int squarewidth=SCREENWIDTH / GRIDWIDTH;
            int squareheight=SCREENHEIGHT / GRIDHEIGHT;
            square[e.getX()/squarewidth][e.getY()/squareheight]=!square[e.getX()/squarewidth][e.getY()/squareheight];
            lastx=e.getX()/squarewidth;
            lasty=e.getY()/squareheight;
            repaint();
        }
        else
            drawingCompleted();
    }
    public void mouseReleased(MouseEvent e) { }
    public void mouseClicked(MouseEvent e) { }
    public void mouseEntered(MouseEvent e) { }
    public void mouseExited(MouseEvent e) { }

    private int lastx=-1,lasty=-1;
    public void mouseDragged(MouseEvent e)
    {
        int squarewidth=SCREENWIDTH / GRIDWIDTH;
        int squareheight=SCREENHEIGHT / GRIDHEIGHT;
        if(lastx==e.getX()/squarewidth && lasty==e.getY()/squareheight) return;
//		square[e.getX()/squarewidth][e.getY()/squareheight]=!square[e.getX()/squarewidth][e.getY()/squareheight];
        square[e.getX()/squarewidth][e.getY()/squareheight]=true;
        lastx=e.getX()/squarewidth;
        lasty=e.getY()/squareheight;
        repaint();
    }
    public void mouseMoved(MouseEvent e) { }

    public static void printUsage()
    {
        System.out.println("Usage:");
        System.out.println(" java OCR sample A");
        System.out.println(" java OCR train");
        System.out.println(" java OCR test");
    }
    public static void main(String[] args)
    {
        if(args.length<1)
            printUsage();
        else if(args[0].equals("sample"))
        {
            new OCR(SAMPLE,args[1].charAt(0));
        }
        else if(args[0].equals("train"))
            new OCR(TRAIN);
        else if(args[0].equals("test"))
            new OCR(TEST);
        else
            printUsage();
    }
}