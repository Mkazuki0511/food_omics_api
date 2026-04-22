import os
import json
import pubchempy as pcp  # 💡 Colab仕様の強力なライブラリを追加！
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 あなたのGemini APIキー
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 既知成分データベース
KNOWN_DB = {
    "curcumin": {"smiles": "COC1=C(O)C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(O)C=C2)OC", "target": "Keap1"},
    "sulforaphane": {"smiles": "C=CCCS(=O)CCCCN=C=S", "target": "Keap1"},
    "resveratrol": {"smiles": "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O", "target": "SIRT1"},
}

def get_smiles(name):
    """ローカルDB優先、無ければPubChemPyでSMILESを取得"""
    name_lower = name.lower()
    
    # 1. まずローカルの既知DBをチェック（最速＆確実）
    if name_lower in KNOWN_DB:
        print(f"✅ ローカルDBから {name} を発見！")
        return KNOWN_DB[name_lower]["smiles"]
    
    # 2. PubChemPyで検索（Colab仕様）
    try:
        print(f"🔍 PubChemで {name} を検索中...")
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            print("✅ PubChemからSMILES取得成功！")
            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"❌ PubChem検索エラー: {e}")
        return None
        
    return None

def get_similarity(input_smiles):
    input_mol = Chem.MolFromSmiles(input_smiles)
    if not input_mol: return None
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    input_fp = gen.GetFingerprint(input_mol)
    best = {"name": "", "similarity": 0.0, "target": ""}
    for db_name, data in KNOWN_DB.items():
        db_fp = gen.GetFingerprint(Chem.MolFromSmiles(data["smiles"]))
        sim = DataStructs.TanimotoSimilarity(input_fp, db_fp) * 100
        if sim > best["similarity"]:
            best = {"name": db_name, "similarity": sim, "target": data["target"]}
    return best

@app.get("/api/analyze")
def analyze(name: str):
    # 1. 構造の取得
    smiles = get_smiles(name)
    if not smiles:
        raise HTTPException(status_code=404, detail="成分の構造が見つかりませんでした。英語名で入力してください。")

    # 2. 類似度チェック & Gemini解析
    sim_match = get_similarity(smiles)
    context = f"SMILES構造: {smiles}。"
    if sim_match and sim_match["similarity"] > 30:
        context += f"構造が{sim_match['name']}に{sim_match['similarity']:.1f}%類似しており、{sim_match['target']}への結合が推論されます。"

    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
        あなたはプロの計算生物学者および生化学者です。
        以下の成分名とコンテキストデータに基づき、詳細な解析結果を**必ず以下のJSONフォーマットで**出力してください。
        マークダウン表記（```json など）は一切含めず、純粋なJSON文字列のみを返してください。数字は文字列ではなく数値型（int/float）で出力してください。

        成分名: {name}
        解析データ: {context}

        {{
          "chemical_identity": {{
            "name": "{name}",
            "iupac": "IUPAC名を記述",
            "formula": "分子式を記述",
            "smiles": "SMILES文字列を記述",
            "mw": 177.29,
            "logp": 0.22,
            "description": "この成分の生化学的な特徴と主要な経路への影響を3〜4文で専門的に解説"
          }},
          "targets": [
            {{"name": "最もスコアの高いターゲット名", "score": 95}},
            {{"name": "2番目のターゲット名", "score": 60}},
            {{"name": "3番目のターゲット名", "score": 20}}
          ],
          "interaction": {{
            "pdb_id": "代表的なPDB ID (例: 4IFJ)",
            "mechanism": "ターゲットタンパク質との結合メカニズム、解離や分解への影響を詳細に解説",
            "cys_residues": [
              {{"name": "Cys151", "domain": "BTB Domain", "context": "Sequence Contextを記述..."}},
              {{"name": "Cys273 & Cys288", "domain": "IVR Domain", "context": "Sequence Contextを記述..."}}
            ],
            "sequence": "ターゲットタンパク質の代表的なアミノ酸配列（FASTA形式の文字列）"
          }},
          "applications": [
            {{"title": "Natural Sources", "value": "多く含まれる食品（例: ブロッコリー、ケールなど）"}},
            {{"title": "Bioavailability", "value": "生体内利用効率や吸収に関する特徴"}},
            {{"title": "Safety Profile", "value": "安全性や毒性に関する知見"}},
            {{"title": "Practical Utility", "value": "機能性食品や創薬への応用可能性"}}
          ],
          "references": [
            {{"title": "論文タイトルやデータベースの参考情報1", "url": "URLまたはPMID"}},
            {{"title": "論文タイトルやデータベースの参考情報2", "url": "URLまたはPMID"}}
          ]
        }}
    """
    
    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace('```json\n', '').replace('```', '').strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"💥 エラー詳細: {e}")
        
        # 💡 API制限時は、Flutterがエラーにならないように完璧なダミーデータを返す
        print("⚠️ APIエラー検知！Flutterの画面テスト用ダミーデータを返します。")
        return {
          "chemical_identity": {
            "name": name if name else "Sulforaphane",
            "iupac": "1-isothiocyanato-4-(methylsulfinyl)butane",
            "formula": "C6H11NOS2",
            "smiles": "CS(=O)CCCCN=C=S",
            "mw": 177.29,
            "logp": 0.22,
            "description": "[API制限中のテストデータ] Sulforaphane acts as an electrophile that covalently modifies specific cysteine residues of target proteins through Michael addition."
          },
          "targets": [
            {"name": "KEAP1", "score": 95},
            {"name": "Tubulin", "score": 60},
            {"name": "HMGB1", "score": 20}
          ],
          "interaction": {
            "pdb_id": "4IFJ",
            "mechanism": "Sulforaphane covalently modifies Cys151 in the BTB domain of KEAP1, leading to conformational changes that disrupt Keap1-mediated ubiquitination and degradation of Nrf2.",
            "cys_residues": [
              {"name": "Cys151", "domain": "BTB Domain", "context": "...148-KHEV C 151 EHQE-154..."},
              {"name": "Cys273 & Cys288", "domain": "IVR Domain", "context": "...268-CEIL C 273 YPGC-277..."}
            ],
            "sequence": ">sp|Q14145|KEAP1_HUMAN\nMQPDPRPSGAGACCRFLPLQSQCPEGAGDAVMYASTECKAEVTPSQHGNRTFSYTLEDHTK..."
          },
          "applications": [
            {"title": "Natural Sources", "value": "Broccoli sprouts, kale, cabbage"},
            {"title": "Bioavailability", "value": "High absorption rate when converted from glucoraphanin."},
            {"title": "Safety Profile", "value": "Generally recognized as safe (GRAS)."},
            {"title": "Practical Utility", "value": "Strong potential for functional foods targeting antioxidant support."}
          ],
          "references": [
            {"title": "Keap1-Nrf2 pathway in health and disease", "url": "PMID: 12345678"},
            {"title": "Structural basis of Keap1 interactions", "url": "PDB: 4IFJ"}
          ]
        }
