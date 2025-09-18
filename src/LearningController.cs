using System;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace WeArt.Components
{
    [System.Serializable]
    public class JointEulerAngles
    {
        public Vector3[] jointEulerAngles = new Vector3[15]; // 5 dita × 3 articolazioni
    }

    public class LearningController : MonoBehaviour
    {
        [Header("Neural Hand Control")]
        public Transform thumb1;
        public Transform thumb2;
        public Transform thumb3;

        public Transform index1;
        public Transform index2;
        public Transform index3;

        public Transform middle1;
        public Transform middle2;
        public Transform middle3;

        public Transform ring1;
        public Transform ring2;
        public Transform ring3;

        public Transform pinky1;
        public Transform pinky2;
        public Transform pinky3;

        public Transform palmTransform;

        [Header("WeArt Thimbles")]
        [SerializeField] private WeArtThimbleTrackingObject thumbThimble;
        [SerializeField] private WeArtThimbleTrackingObject indexThimble;
        [SerializeField] private WeArtThimbleTrackingObject middleThimble;

        [Header("Socket Settings")]
        public string serverIP = "127.0.0.1";
        public int serverPort = 65432;

        [Header("Predicted Joint Angles (Read-Only)")]
        public JointEulerAngles debugJointAngles = new JointEulerAngles();

        [Header("Real-Time Input Debug")]
        public float thumbClosure;
        public float indexClosure;
        public float middleClosure;
        public float thumbAbduction;

        private TcpClient client;
        private NetworkStream stream;
        private Thread socketThread;
        private float[] closureAbductionInput = new float[4]; // [thumbC, indexC, middleC, abduction]
        private float[] receivedJointAngles = new float[45];  // 15 articolazioni × 3 angoli di Eulero
        private bool newDataAvailable = false;
        private bool isRunning = true;

        [Header("Keyboard Simulation")]
        [Tooltip("How quickly the simulated input values change when a key is pressed.")]
        public float simulationSpeed = 5.0f;

        void Start()
        {
            // Assegna i Transform delle articolazioni utilizzando i percorsi nella gerarchia
            //thumb1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R");
            //thumb2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R");
            //thumb3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R");

            //index1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R");
            //index2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R");
            //index3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R/DEF-f_index.03.R");

            //middle1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R");
            //middle2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R");
            //middle3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R/DEF-f_middle.03.R");

            //ring1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R");
            //ring2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R");
            //ring3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R/DEF-f_ring.03.R");

            //pinky1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R");
            //pinky2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R");
            //pinky3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R/DEF-f_pinky.03.R");

            //palmTransform = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R");

            // Avvia il thread per la comunicazione socket
            socketThread = new Thread(SocketLoop);
            socketThread.IsBackground = true;
            socketThread.Start();
        }

        void Update()
        {
            // Input in tempo reale dai thimble WeArt
            if (thumbThimble != null && indexThimble != null && middleThimble != null)
            {
                //closureAbductionInput[0] = thumbThimble.Closure.Value;
                //closureAbductionInput[1] = indexThimble.Closure.Value;
                //closureAbductionInput[2] = middleThimble.Closure.Value;
                //closureAbductionInput[3] = thumbThimble.Abduction.Value;

                //closureAbductionInput[0] = Input.GetKey(KeyCode.T) ? 1.0f : 0.0f; // Simula chiusura pollice
                //closureAbductionInput[1] = Input.GetKey(KeyCode.I) ? 1.0f : 0.0f; // Simula chiusura indice
                //closureAbductionInput[2] = Input.GetKey(KeyCode.M) ? 1.0f : 0.0f; // Simula chiusura medio
                //closureAbductionInput[3] = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f; // Simula abduzione pollice
                
                // 1. Determine the target value for each input (1.0 if pressed, 0.0 if not)
                float targetThumbClosure = Input.GetKey(KeyCode.T) ? 1.0f : 0.0f;
                float targetIndexClosure = Input.GetKey(KeyCode.I) ? 1.0f : 0.0f;
                float targetMiddleClosure = Input.GetKey(KeyCode.M) ? 1.0f : 0.0f;
                float targetThumbAbduction = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;

                // 2. Smoothly move the current input value towards the target value over time
                closureAbductionInput[0] = Mathf.Lerp(closureAbductionInput[0], targetThumbClosure, Time.deltaTime * simulationSpeed);
                closureAbductionInput[1] = Mathf.Lerp(closureAbductionInput[1], targetIndexClosure, Time.deltaTime * simulationSpeed);
                closureAbductionInput[2] = Mathf.Lerp(closureAbductionInput[2], targetMiddleClosure, Time.deltaTime * simulationSpeed);
                closureAbductionInput[3] = Mathf.Lerp(closureAbductionInput[3], targetThumbAbduction, Time.deltaTime * simulationSpeed);

                // Aggiorna i valori per l'Inspector
                thumbClosure = closureAbductionInput[0];
                indexClosure = closureAbductionInput[1];
                middleClosure = closureAbductionInput[2];
                thumbAbduction = closureAbductionInput[3];
            }

            // Applica le rotazioni previste alle articolazioni
            if (newDataAvailable)
            {
                Transform[] joints = new Transform[]
                {
                    thumb1, thumb2, thumb3,
                    index1, index2, index3,
                    middle1, middle2, middle3,
                    ring1, ring2, ring3,
                    pinky1, pinky2, pinky3
                };

                for (int i = 0; i < joints.Length; i++)
                {
                    if (joints[i] != null)
                    {
                        int index = i * 3;
                        Vector3 predictedEuler = new Vector3(
                            receivedJointAngles[index],
                            receivedJointAngles[index + 1],
                            receivedJointAngles[index + 2]
                        );

                        joints[i].rotation = palmTransform.rotation * Quaternion.Euler(predictedEuler);

                        // Salva per la visualizzazione nell'Inspector
                        debugJointAngles.jointEulerAngles[i] = predictedEuler;
                    }
                }
                newDataAvailable = false;
            }
        }

        void SocketLoop()
        {
            try
            {
                client = new TcpClient(serverIP, serverPort);
                stream = client.GetStream();
                Debug.Log("Connesso al server Python.");

                byte[] inputBuffer = new byte[4 * 4];   // 4 float
                byte[] outputBuffer = new byte[45 * 4]; // 45 float

                while (isRunning)
                {
                    // Prepara l'input: 4 float → array di byte
                    Buffer.BlockCopy(closureAbductionInput, 0, inputBuffer, 0, inputBuffer.Length);
                    stream.Write(inputBuffer, 0, inputBuffer.Length);

                    // Legge tutti i 180 byte (45 float)
                    int totalRead = 0;
                    while (totalRead < outputBuffer.Length)
                    {
                        int bytesRead = stream.Read(outputBuffer, totalRead, outputBuffer.Length - totalRead);
                        if (bytesRead == 0) throw new Exception("Disconnesso dal server.");
                        totalRead += bytesRead;
                    }

                    // Converte in array di float
                    float[] predictions = new float[45];
                    Buffer.BlockCopy(outputBuffer, 0, predictions, 0, outputBuffer.Length);
                    receivedJointAngles = predictions;
                    newDataAvailable = true;
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Errore socket: " + e.Message);
            }
        }

        void OnApplicationQuit()
        {
            isRunning = false;
            stream?.Close();
            client?.Close();
            socketThread?.Abort();
        }
    }
}
