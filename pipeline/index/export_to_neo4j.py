"""
将规范化后的三元组导出到 Neo4j
必须先运行 build_index.py 生成 graphrag_index/merged_triples.json

运行: python export_to_neo4j.py
"""

import json, os
from pathlib import Path

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")  # ← 改这里
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# 直接读 build_index.py 生成的规范化文件
TRIPLES_PATH = Path("graphrag_index/merged_triples.json")

TYPE_LABEL = {
    "RadarSystem":      "Radar",    "Radar":            "Radar",
    "Manufacturer":     "Company",  "Company":          "Company",
    "Organization":     "Organization",
    "Platform":         "Platform", "NavalVessel":      "Platform",
    "AircraftPlatform": "Platform", "GroundPlatform":   "Platform",
    "Aircraft":         "Platform",
    "Country":          "Country",
    "FrequencyBand":    "FrequencyBand",
    "Function":         "Function",
    # 新增类型
    "Weapon":           "Weapon",   "Missile":          "Weapon",
    "RadarMode":        "RadarMode",
}
PLATFORM_SUBTYPE = {
    "NavalVessel": "naval", "AircraftPlatform": "aircraft", "GroundPlatform": "ground",
}

def get_label(t):   return TYPE_LABEL.get(t, "Entity")
def get_subtype(t): return PLATFORM_SUBTYPE.get(t, "")

def export_to_neo4j(triples):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        print("清空旧数据...")
        session.run("MATCH (n) DETACH DELETE n")

        print("创建唯一约束...")
        for label in ["Radar","Company","Country","Platform",
                      "FrequencyBand","Organization","Function",
                      "Weapon","RadarMode","Entity"]:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                f"REQUIRE n.name IS UNIQUE"
            )

        BATCH, total = 200, len(triples)
        print(f"导入 {total} 条三元组...")

        for i in range(0, total, BATCH):
            batch = triples[i: i+BATCH]
            with session.begin_transaction() as tx:
                for t in batch:
                    head, relation, tail = (
                        t.get("head","").strip(),
                        t.get("relation","").strip(),
                        t.get("tail","").strip()
                    )
                    if not head or not relation or not tail:
                        continue
                    h_type, t_type = t.get("head_type",""), t.get("tail_type","")
                    h_label, t_label = get_label(h_type), get_label(t_type)

                    h_props = {"name": head, "entity_type": h_type}
                    if h_label == "Platform":
                        h_props["platform_type"] = get_subtype(h_type)
                    tx.run(f"MERGE (n:{h_label} {{name:$name}}) SET n+=$props",
                           name=head, props=h_props)

                    t_props = {"name": tail, "entity_type": t_type}
                    if t_label == "Platform":
                        t_props["platform_type"] = get_subtype(t_type)
                    tx.run(f"MERGE (n:{t_label} {{name:$name}}) SET n+=$props",
                           name=tail, props=t_props)

                    tx.run(
                        f"MATCH (h:{h_label} {{name:$head}}) "
                        f"MATCH (t:{t_label} {{name:$tail}}) "
                        f"MERGE (h)-[r:{relation}]->(t) "
                        f"SET r.confidence=$conf, r.evidence=$ev, r.source=$src",
                        head=head, tail=tail,
                        conf=float(t.get("confidence",0.8)),
                        ev=t.get("evidence","")[:300],
                        src=t.get("source","")
                    )
                tx.commit()
            print(f"  {min(i+BATCH,total)}/{total}...")

    driver.close()

def print_stats():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        nc = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rc = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\n图数据库: {nc} 节点, {rc} 关系")
        print("\n节点类型:")
        for label in ["Radar","Company","Country","Platform",
                      "FrequencyBand","Organization","Function",
                      "Weapon","RadarMode","Entity"]:
            n = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            if n: print(f"  {label}: {n}")
        print("\n关系类型:")
        for row in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS c ORDER BY c DESC"
        ):
            print(f"  {row['rel']}: {row['c']}")
    driver.close()

if __name__ == "__main__":
    if not TRIPLES_PATH.exists():
        print("❌ 找不到 graphrag_index/merged_triples.json")
        print("   请先运行: python build_index.py")
        exit(1)

    print(f"读取规范化三元组: {TRIPLES_PATH}")
    with open(TRIPLES_PATH, encoding="utf-8") as f:
        triples = json.load(f)
    print(f"共 {len(triples)} 条（已规范化）\n")

    export_to_neo4j(triples)
    print_stats()

    print("\n常用 Cypher 查询:")
    queries = [
        ("全图预览",      "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100"),
        ("某雷达所有关系", "MATCH (r:Radar {name:'AN/TPY-2'})-[rel]->(n) RETURN r,rel,n"),
        ("Raytheon研制",  "MATCH (c:Company {name:'Raytheon'})<-[:developedBy]-(r:Radar) RETURN r,c"),
        ("X波段雷达",     "MATCH (r:Radar)-[:hasFrequencyBand]->(f:FrequencyBand {name:'X'}) RETURN r,f"),
        ("各国装备数量",  "MATCH (c:Country)<-[:operatedBy]-(r:Radar) RETURN c.name,count(r) AS n ORDER BY n DESC"),
        ("制造商排名",    "MATCH (c:Company)<-[:developedBy]-(r:Radar) RETURN c.name,count(r) AS n ORDER BY n DESC LIMIT 10"),
    ]
    for name, q in queries:
        print(f"  # {name}\n  {q}\n")
