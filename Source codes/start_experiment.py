import sys
import os

# 將子目錄加入路徑以確保能導入
sys.path.append(os.path.abspath('badnets'))

# 我們將模擬 badnets_cifar_fedavg 的執行環境，但修正 Windows 多進程 Bug
def main():
    import badnets_cifar_fedavg as experiment
    
    # 覆寫全局變數為短周期測試
    experiment.ROUNDS = 2
    
    # 手動執行實驗的主迴圈邏輯 (基於 badnets/badnets_cifar_fedavg.py)
    print("--- STARTING BADNETS CIFAR-10 FEDAVG TEST (2 ROUNDS) ---")
    
    for r in range(1, experiment.ROUNDS + 1):
        local_states = []
        for cid in range(experiment.NUM_CLIENTS):
            local_model = experiment.copy.deepcopy(experiment.model_global)
            epochs = experiment.LOCAL_EPOCHS_ATTACKER if cid == experiment.ATTACKER_IDX else experiment.LOCAL_EPOCHS_BENIGN
            experiment.train_local(local_model, experiment.client_loaders[cid], epochs=epochs)
            st = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
            local_states.append(st)
        
        experiment.fedavg(experiment.model_global, local_states)
        acc = experiment.eval_acc(experiment.model_global, experiment.test_loader)
        asr = experiment.eval_asr(experiment.model_global, experiment.test_loader)
        print(f"Round {r:02d} | Clean ACC = {acc:.2f}% | ASR = {asr*100:.2f}%")

if __name__ == '__main__':
    main()
