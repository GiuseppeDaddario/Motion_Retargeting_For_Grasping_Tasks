using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Globalization;
using System.Collections.Generic;
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
        public Transform thumb1, thumb2, thumb3;
        public Transform index1, index2, index3;
        public Transform middle1, middle2, middle3;
        public Transform ring1, ring2, ring3;
        public Transform pinky1, pinky2, pinky3;
        public Transform palmTransform;

        [Header("Socket Settings")]
        public string serverIP = "127.0.0.1";
        public int serverPort = 65432;

        [Header("CSV Dataset")]
        [Tooltip("Nome del file CSV, deve stare nella stessa cartella dello script all'interno di Assets.")]
        public string csvFileName = "dataset.csv";

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

        private List<float[]> testDataset = new List<float[]>();
        private int currentTestIndex = 0;

        private float[] closureAbductionInput = new float[4];
        private float[] receivedJointAngles = new float[45];
        private bool newDataAvailable = false;
        private volatile bool isRunning = true;

        void Start()
        {
            // Assegna i Transform delle articolazioni (puoi adattare ai tuoi percorsi)
            thumb1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R");
            thumb2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R");
            thumb3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R");

            index1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R");
            index2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R");
            index3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.01.R/DEF-f_index.01.R/DEF-f_index.02.R/DEF-f_index.03.R");

            middle1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R");
            middle2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R");
            middle3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.02.R/DEF-f_middle.01.R/DEF-f_middle.02.R/DEF-f_middle.03.R");

            ring1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R");
            ring2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R");
            ring3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.03.R/DEF-f_ring.01.R/DEF-f_ring.02.R/DEF-f_ring.03.R");

            pinky1 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R");
            pinky2 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R");
            pinky3 = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R/ORG-palm.04.R/DEF-f_pinky.01.R/DEF-f_pinky.02.R/DEF-f_pinky.03.R");

            palmTransform = transform.Find("Hands/WEARTLeftHand/HandRig/HandRoot/DEF-hand.R");

            // Carica il dataset CSV
            LoadTestDataset();

            // Avvia thread socket
            socketThread = new Thread(SocketLoop);
            socketThread.IsBackground = true;
            socketThread.Start();
        }

        void Update()
        {
            // Avanza al prossimo campione del dataset con barra spaziatrice
            if (Input.GetKeyDown(KeyCode.Space))
            {
                ApplyNextTestPose();
            }

            // Applica la posa ricevuta dal server
            if (newDataAvailable)
            {
                ApplyReceivedPose();
                newDataAvailable = false;
            }
        }

        private void LoadTestDataset()
        {
            string fullPath = Path.Combine(Application.streamingAssetsPath, "dataset.csv");

            if (!File.Exists(fullPath))
            {
                Debug.LogError("CSV non trovato: " + fullPath);
                return;
            }

            testDataset.Clear();
            string[] lines = File.ReadAllLines(fullPath);

            for (int i = 1; i < lines.Length; i++) // salta header
            {
                string[] values = lines[i].Split(',');
                if (values.Length < 4) continue;

                float thumb = float.Parse(values[values.Length - 4], CultureInfo.InvariantCulture);
                float index = float.Parse(values[values.Length - 3], CultureInfo.InvariantCulture);
                float middle = float.Parse(values[values.Length - 2], CultureInfo.InvariantCulture);
                float abduction = float.Parse(values[values.Length - 1], CultureInfo.InvariantCulture);

                testDataset.Add(new float[] { thumb, index, middle, abduction });
            }

            Debug.Log("Dataset caricato: " + testDataset.Count + " campioni.");
        }

        private void ApplyNextTestPose()
        {
            if (testDataset.Count == 0) return;

            closureAbductionInput = testDataset[currentTestIndex];

            thumbClosure = closureAbductionInput[0];
            indexClosure = closureAbductionInput[1];
            middleClosure = closureAbductionInput[2];
            thumbAbduction = closureAbductionInput[3];

            currentTestIndex = (currentTestIndex + 1) % testDataset.Count;
        }

        private void ApplyReceivedPose()
        {
            Transform[] joints = { thumb1, thumb2, thumb3, index1, index2, index3, middle1, middle2, middle3,
                                   ring1, ring2, ring3, pinky1, pinky2, pinky3 };

            for (int i = 0; i < joints.Length; i++)
            {
                if (joints[i] != null)
                {
                    int index = i * 3;
                    Vector3 predictedEuler = new Vector3(receivedJointAngles[index], receivedJointAngles[index + 1], receivedJointAngles[index + 2]);
                    joints[i].rotation = palmTransform.rotation * Quaternion.Euler(predictedEuler);
                    debugJointAngles.jointEulerAngles[i] = predictedEuler;
                }
            }
        }

        private void SocketLoop()
        {
            try
            {
                client = new TcpClient(serverIP, serverPort);
                stream = client.GetStream();
                Debug.Log("Connesso al server Python.");

                byte[] inputBuffer = new byte[4 * 4];
                byte[] outputBuffer = new byte[45 * 4];

                while (isRunning)
                {
                    // Invio dati al server
                    Buffer.BlockCopy(closureAbductionInput, 0, inputBuffer, 0, inputBuffer.Length);
                    stream.Write(inputBuffer, 0, inputBuffer.Length);

                    // Ricezione dati dal server
                    int totalRead = 0;
                    while (totalRead < outputBuffer.Length)
                    {
                        int bytesRead = stream.Read(outputBuffer, totalRead, outputBuffer.Length - totalRead);
                        if (bytesRead == 0) throw new Exception("Disconnesso dal server.");
                        totalRead += bytesRead;
                    }

                    Buffer.BlockCopy(outputBuffer, 0, receivedJointAngles, 0, outputBuffer.Length);
                    newDataAvailable = true;

                    // --- NUOVO: log dei dati ricevuti ---
                    Debug.Log($"Dati ricevuti dal server: Thumb1={receivedJointAngles[0]:F2}, Index1={receivedJointAngles[3]:F2}, Middle1={receivedJointAngles[6]:F2}");
                }
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError("Errore socket: " + e.Message);
                }
            }
            finally
            {
                stream?.Close();
                client?.Close();
            }
        }

        void OnApplicationQuit()
        {
            isRunning = false;
            client?.Close();
            socketThread?.Join(500);
        }
    }
}