import tensorflow as tf
import numpy as np
import pandas as pd
from models.kernel_predictor import MultiKernelPredictor, train_step, analyze_kernel_importance
import matplotlib.pyplot as plt
import os

def train_model(train_data, val_data, kernel_sizes, 
                epochs=100, batch_size=32, 
                learning_rate=0.001,
                save_dir='../results'):
    
    # 모델 및 옵티마이저 초기화
    model = MultiKernelPredictor(kernel_sizes=kernel_sizes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 결과 저장을 위한 디렉토리 생성
    os.makedirs(f"{save_dir}/kernel_importance", exist_ok=True)
    os.makedirs(f"{save_dir}/loss_plots", exist_ok=True)
    
    # 학습 기록
    train_losses = []
    val_losses = []
    kernel_importance_history = []
    
    for epoch in range(epochs):
        # Training
        epoch_train_losses = []
        for batch in train_data.batch(batch_size):
            inputs, targets = batch
            result = train_step(model, inputs, targets, optimizer)
            epoch_train_losses.append(result['loss'])
            
            # 커널 중요도 분석 (배치당 한번)
            if len(kernel_importance_history) % 100 == 0:
                importance_info = analyze_kernel_importance(result['predictions'])
                kernel_importance_history.append({
                    'epoch': epoch,
                    'batch': len(kernel_importance_history),
                    'importance': importance_info
                })
        
        # Validation
        val_epoch_losses = []
        for batch in val_data.batch(batch_size):
            inputs, targets = batch
            val_result = model(inputs, training=False)
            val_loss = tf.reduce_mean(tf.square(targets - val_result['predictions']))
            val_epoch_losses.append(val_loss)
        
        # 에폭별 평균 손실 계산
        avg_train_loss = tf.reduce_mean(epoch_train_losses)
        avg_val_loss = tf.reduce_mean(val_epoch_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 에폭별 결과 출력
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 주기적으로 커널 중요도 시각화
        if (epoch + 1) % 10 == 0:
            plot_kernel_importance(kernel_importance_history[-1]['importance'], 
                                save_path=f"{save_dir}/kernel_importance/epoch_{epoch+1}.png")
    
    # 최종 학습 결과 저장
    plot_training_history(train_losses, val_losses, 
                         save_path=f"{save_dir}/loss_plots/training_history.png")
    save_kernel_importance_history(kernel_importance_history, 
                                 save_path=f"{save_dir}/kernel_importance_history.csv")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'kernel_importance_history': kernel_importance_history
    }

def plot_kernel_importance(importance_info, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(importance_info['kernel_sizes'], importance_info['importance_scores'])
    plt.title('Kernel Importance Analysis')
    plt.xlabel('Kernel Size')
    plt.ylabel('Importance Score')
    plt.savefig(save_path)
    plt.close()

def plot_training_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_kernel_importance_history(history, save_path):
    df = pd.DataFrame([{
        'epoch': h['epoch'],
        'batch': h['batch'],
        'kernel_sizes': h['importance']['kernel_sizes'],
        'importance_scores': h['importance']['importance_scores'],
        'most_important_kernel': h['importance']['most_important_kernel']
    } for h in history])
    df.to_csv(save_path, index=False)
