const DDRAGON = "https://ddragon.leagueoflegends.com";

let versionCache: string | null = null;
let champIdCache: Record<string, string> | null = null;

export async function getDdragonVersion(): Promise<string> {
  if (versionCache) return versionCache;
  const res = await fetch(`${DDRAGON}/api/versions.json`);
  const versions: string[] = await res.json();
  versionCache = versions[0];
  return versionCache;
}

/** Map OE champion names -> Data Dragon image keys */
const ALIASES: Record<string, string> = {
  Wukong: "MonkeyKing",
  "Renata Glasc": "Renata",
  Nunu: "Nunu",
  "Nunu & Willump": "Nunu",
  "Dr. Mundo": "DrMundo",
  "Lee Sin": "LeeSin",
  "Jarvan IV": "JarvanIV",
  "Master Yi": "MasterYi",
  "Miss Fortune": "MissFortune",
  "Twisted Fate": "TwistedFate",
  "Xin Zhao": "XinZhao",
  "Aurelion Sol": "AurelionSol",
  "Tahm Kench": "TahmKench",
  "Bel'Veth": "Belveth",
  "Cho'Gath": "Chogath",
  "Kai'Sa": "Kaisa",
  "Kha'Zix": "Khazix",
  "Kog'Maw": "KogMaw",
  "Rek'Sai": "RekSai",
  "Vel'Koz": "Velkoz",
  "K'Sante": "KSante",
};

export function toDdragonId(name: string): string {
  if (ALIASES[name]) return ALIASES[name];
  return name.replace(/['. ]/g, "").replace(/&/g, "");
}

export async function championIconUrl(name: string): Promise<string> {
  const version = await getDdragonVersion();
  return `${DDRAGON}/cdn/${version}/img/champion/${toDdragonId(name)}.png`;
}

export function championIconUrlSync(name: string, version: string): string {
  return `${DDRAGON}/cdn/${version}/img/champion/${toDdragonId(name)}.png`;
}

export async function loadChampionIdMap(): Promise<Record<string, string>> {
  if (champIdCache) return champIdCache;
  const version = await getDdragonVersion();
  const res = await fetch(`${DDRAGON}/cdn/${version}/data/en_US/champion.json`);
  const data = await res.json();
  const map: Record<string, string> = {};
  for (const champ of Object.values(data.data) as Array<{ id: string; name: string }>) {
    map[champ.name] = champ.id;
    map[champ.id] = champ.id;
  }
  champIdCache = map;
  return map;
}
