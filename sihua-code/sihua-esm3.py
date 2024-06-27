import os
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# 从环境变量中获取token
token = os.getenv('HUGGINGFACE_HUB_TOKEN')  # 确保这里的环境变量名称与SLURM脚本中设置的相匹配

# 使用token进行登录
if token:
    login(token=token)
else:
    raise ValueError("Hugging Face Hub token not found in environment variables.")

# 下载模型权重并实例化模型
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"

# 定义部分蛋白质序列的提示（用于生成完整序列）
#prompt = "_____RSRNRDRRYDEVPSDLPYQDTTIRTHPTLHDSERAVSADPLPPPPLPLQPPFGPDFYSSDTEEPAIAPDLKPVRRFVPDSWKNFFRGKKKDPEWDKPVSDIRYISDGVECSPPASPARPNHRSPLNSCKDPYGGSEGTFSSRKEADAVFPRDPYGSLDRHTQTVRTYSEKVEEYNLRYSYMKSWAGLLRILGVVELLLGAGVFACVTAYIHKDSEWYNLFGYSQPYGMGGVGGLGSMYGGYYYTGPKTPFVLVVAGLAWITTIIILVLGMSMYYRTILLDSNWWPLTEFGINVALFILYMAAAIVYVNDTNRGGLCYYPLFNTPVNAVFCRVEGGQIAAMIFLFVTMIVYLISALVCLKLWRHEAARRHREYMEQQEINEPSLSSKRKMCEMATSGDRQRDSEVNFKELRTAKMKPELLSGHIPPGHIPKPIVMPDYVAKYPVIQTDDERERYKAVFQDQFSEYKELSAEVQAVLRKFDELDAVMSRLPHHSESRQEHERISRIHEEFKKKKNDPTFLEKKERCDYLKNKLSHIKQRIQEYDKVMNWD_____"
prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
protein = ESMProtein(sequence=prompt)

# 生成序列，然后是结构。这将迭代地揭示序列轨迹。
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# 我们可以显示生成序列的预测结构。
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./generation-3.pdb")

# 然后我们可以通过逆折叠序列并重新计算结构进行回路设计
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./round_tripped-3.pdb")

