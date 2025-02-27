# **Heart Disease Prediction - End-to-End MLOps Project**  

## **📌 Project Description**
This project builds and deploys a **Heart Disease Prediction API** using **Machine Learning (ML) and MLOps best practices**. The goal is to demonstrate **end-to-end ML production skills** by implementing:

- ✅ **Data Preprocessing & Model Training**: Random Forest, XGBoost, and Neural Networks.
- ✅ **Experiment Tracking & Model Logging**: MLflow to track model performance.
- ✅ **Containerization & Deployment**: Dockerized API deployed on AWS EC2.
- ✅ **CI/CD Pipeline**: Automated deployment using GitHub Actions.

This project ensures that every change is **automatically built, tested, and deployed** to AWS EC2 via **CI/CD**.

---

## **📌 Features & MLOps Workflow**
✅ **End-to-End ML Pipeline**: Data processing → Model Training → Deployment  
✅ **Model Tracking with MLflow** (Track Accuracy, AUC, Hyperparameters)  
✅ **Containerized API with Docker** (Runs on Any System)  
✅ **CI/CD Pipeline with GitHub Actions** (Auto Deploys on Push)  
✅ **Deployed on AWS EC2** (Live API Ready)  

---

## **📌 Deployment Details**
- **Docker Hub**: The API is available as a Docker image at: 
  - `docker pull fengdizhao/heart-disease-api:latest`
- **AWS EC2**: The API is running on a public EC2 instance (`44.210.111.71`)
- **CI/CD Pipeline**: GitHub Actions automates deployment on every push to `main`.

## **📌 How to Use the API**

---

### **1️⃣ Test API Locally**
If running locally, use the following cURL command:
```bash
curl -X POST "http://127.0.0.1:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}'
```

### **2️⃣ Test API on AWS EC2**
The API is publicly available at **http://44.210.111.71:5001**

```bash
curl -X POST "http://44.210.111.71:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}'
```

---

## **📌 Model Performance**
✅ **Neural Network performed best with:**
- **Accuracy: 0.88**
- **AUC: 0.88**

---

## **📌 Next Steps & Future Improvements**
🚀 **Planned Enhancements:**
- Automate **model retraining** when new data arrives.
- Implement **authentication & API security**.
- Deploy on **Kubernetes (K8s) for scalability**.
- Integrate **real-time monitoring** with Prometheus/Grafana.

---

## **🎯 Conclusion**
This project successfully demonstrates **end-to-end MLOps practices** by:

✅ **Building & Training ML Models**  
✅ **Tracking Model Performance with MLflow**  
✅ **Deploying an API on AWS EC2**  
✅ **Automating CI/CD with GitHub Actions**  

