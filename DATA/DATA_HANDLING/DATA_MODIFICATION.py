import os
import pandas as pd
import numpy as np

OUTPUT_DATASET = 'DATA/FULL_FEATURE_SET/COMBINED_DATASET.csv'
ORIGINAL_DATASET_DIRECTORY = 'DATA/ORIGINAL'
NODEHEX_MAPPING = {
    "12DA0": {"latitude": 51.89796189901707, "longitude": -8.34958654828051},
    "363A6": {"latitude": 51.90151784278236, "longitude": -8.467069184461492},
    "3640A": {"latitude": 51.89824636047899, "longitude": -8.444166971365396},
    "36536": {"latitude": 51.898795777138986, "longitude": -8.465151582292544},
    "3D160": {"latitude": 51.87626038669227, "longitude": -8.549984144793676},
    "4324A": {"latitude": 51.87781810784789, "longitude": -8.433045682768261},
    "43300": {"latitude": 51.89431459159797, "longitude": -8.47493724402924},
    "43376": {"latitude": 51.897431848557105, "longitude": -8.477237029621136},
    "43506": {"latitude": 51.90351928706315, "longitude": -8.497545319261157},
    "43696": {"latitude": 51.886950697539184, "longitude": -8.409477962957487},
    "438EE": {"latitude": 51.92230774476771, "longitude": -8.424083830316144},
    "5328A": {"latitude": 51.90345608935438, "longitude": -8.479636588775353},
    "532EE": {"latitude": 51.9110518184705, "longitude": -8.461365518262927},
    "534E2": {"latitude": 51.91226978477891, "longitude": -8.439918438171041},
    "53802": {"latitude": 51.92342778113091, "longitude": -8.487211515581768},
    "53B22": {"latitude": 51.89369373071137, "longitude": -8.508297113386531},
    "53BEA": {"latitude": 51.89837232554428, "longitude": -8.49658697357854},
    "53C4E": {"latitude": 51.891675002809286, "longitude": -8.454116230255268},
    "53E42": {"latitude": 51.88628594297836, "longitude": -8.459255380962231},
    "53FD2": {"latitude": 51.908336489003425, "longitude": -8.489225296020251},
    "540FE": {"latitude": 51.894624443856735, "longitude": -8.445768716620313},
    "542F2": {"latitude": 51.904327027255455, "longitude": -8.488935454420005},
    "5A5BC": {"latitude": 51.89778159615715, "longitude": -8.482514817473437},
    "5A68E": {"latitude": 51.9028034923363, "longitude": -8.47443862283969},
    "5A6F2": {"latitude": 51.89661057113405, "longitude": -8.468960998015731},
    "5A756": {"latitude": 51.90413682754215, "longitude": -8.423470116319923},
    "5A7BA": {"latitude": 51.89100026121953, "longitude": -8.490063964938848},
    "5A8": {"latitude": 51.891038858626956, "longitude": -8.490076994196404},
    "5A81E": {"latitude": 51.89077136248385, "longitude": -8.463134781010858},
    "5AC06": {"latitude": 51.91267625011353, "longitude": -8.490395518846482},
    "5ACCE": {"latitude": 51.89807732789838, "longitude": -8.469114442201398},
    "613E4": {"latitude": 51.90874708559133, "longitude": -8.455577926931698},
    "613EE": {"latitude": 51.90874708559133, "longitude": -8.455577926931698},
    "6157E": {"latitude": 51.89187340679471, "longitude": -8.435898272700415},
    "A271": {"latitude": 51.91171105032543, "longitude": -8.277255310589522},
    "A4DF": {"latitude": 51.88295128827059, "longitude": -8.398523548359371},
    "A4E8": {"latitude": 51.910545102885465, "longitude": -8.473114683421914},
    "A4EF": {"latitude": 51.90427888705044, "longitude": -8.468722963211876},
    "A515": {"latitude": 51.876301674974606, "longitude": -8.313483868475375},
    "A5B0": {"latitude": 51.93343528738021, "longitude": -8.56407333911494},
    "A700": {"latitude": 51.89437653199997, "longitude": -8.474961102453534},
    "A701": {"latitude": 51.89750873856607, "longitude": -8.47727851523633},
    "A705": {"latitude": 51.90356600364942, "longitude": -8.497543975548506},
    "A706": {"latitude": 51.89845997380454, "longitude": -8.47398127196009},
    "A707": {"latitude": 51.896633487225806, "longitude": -8.464480632781436},
    "A709": {"latitude": 51.88693341096669, "longitude": -8.409533589118725},
    "A710": {"latitude": 51.92102525244761, "longitude": -8.474425045701944},
    "A81A": {"latitude": 52.018865102109125, "longitude": -8.305002645099014},
    "A81B": {"latitude": 51.937094759238846, "longitude": -8.391602750837297},
    "A990": {"latitude": 51.90545797523891, "longitude": -8.458922196422025},
    "A992": {"latitude": 51.902198409969245, "longitude": -8.481813344720807},
    "A995": {"latitude": 51.904633032567354, "longitude": -8.450547174916233},
    "A99B": {"latitude": 51.92471393212021, "longitude": -8.413284459772884},
    "A99D": {"latitude": 51.909659912032254, "longitude": -8.414420814738111},
    "A99E": {"latitude": 51.900616505990676, "longitude": -8.3696101979449},
    "A9AA": {"latitude": 51.89787060978882, "longitude": -8.482617011494368},
    "A9B0": {"latitude": 51.90829594861598, "longitude": -8.489080849299103},
    "A9B6": {"latitude": 51.89270536139112, "longitude": -8.41841255382451},
    "A9C9": {"latitude": 51.87180599810642, "longitude": -8.462966121229485},
    "AAB4": {"latitude": 51.87732796871184, "longitude": -8.507830025275178},
    "AAB6": {"latitude": 51.88050430145362, "longitude": -8.471038875729459},
    "AACF": {"latitude": 51.883966863402165, "longitude": -8.48854413554602},
    "AC60": {"latitude": 51.887921647105735, "longitude": -8.583536368412915},
    "C100": {"latitude": 51.890487670898, "longitude": -8.4574127197266},
    "C200": {"latitude": 51.889311, "longitude": -8.461533},
    "C300": {"latitude": 51.889285, "longitude": -8.508105},
    "C400": {"latitude": 51.919128, "longitude": -8.481977},
    "C500": {"latitude": 51.904555, "longitude": -8.382396},
    "C600": {"latitude": 51.887054, "longitude": -8.43132},
    "C700": {"latitude": 51.952042, "longitude": -8.427277},
    "C800": {"latitude": 52.031499, "longitude": -8.60133},
    "C900": {"latitude": 51.924624, "longitude": -8.515091},
    "C901": {"latitude": 51.916237, "longitude": -8.43132},
    "C902": {"latitude": 51.91864, "longitude": -8.399734},
    "C903": {"latitude": 51.952031, "longitude": -8.427249},
    "C904": {"latitude": 51.897348, "longitude": -8.496921},
    "C905": {"latitude": 51.899145, "longitude": -8.396948},
    "C906": {"latitude": 51.89409, "longitude": -8.490147},
    "C907": {"latitude": 51.90271, "longitude": -8.450237},
    "C908": {"latitude": 51.901461, "longitude": -8.396604},
    "C909": {"latitude": 51.897354, "longitude": -8.474236},
    "C910": {"latitude": 51.903511, "longitude": -8.383033},
    "C911": {"latitude": 51.929093, "longitude": -8.527098},
    "C912": {"latitude": 51.932667, "longitude": -8.478798},
    "C913": {"latitude": 51.937184, "longitude": -8.452615},
    "C914": {"latitude": 51.92638, "longitude": -8.489172},
    "C915": {"latitude": 51.875381469727, "longitude": -8.4258270263672},
    "C916": {"latitude": 51.905369, "longitude": -8.389559},
    "C917": {"latitude": 51.908340454102, "longitude": -8.4821319580078},
    "C918": {"latitude": 51.907883, "longitude": -8.395157},
    "C919": {"latitude": 51.91658, "longitude": -8.438645},
    "C920": {"latitude": 51.892548, "longitude": -8.468056},
    "C921": {"latitude": 51.886826, "longitude": -8.444366},
    "C922": {"latitude": 51.902847, "longitude": -8.348923},
    "C923": {"latitude": 51.896667, "longitude": -8.488998},
    "C924": {"latitude": 51.890487670898, "longitude": -8.4574127197266},
    "C925": {"latitude": 51.898727416992, "longitude": -8.4738922119141},
    "C926": {"latitude": 51.900101, "longitude": -8.464554},
    "C927": {"latitude": 51.921559, "longitude": -8.465481},
    "C928": {"latitude": 51.914932, "longitude": -8.464554},
    "C929": {"latitude": 51.916580200195, "longitude": -8.4793853759766},
    "C930": {"latitude": 51.91520690918, "longitude": -8.4766387939453},
    "C931": {"latitude": 51.893692, "longitude": -8.467941},
    "C932": {"latitude": 51.895981, "longitude": -8.47435},
    "C933": {"latitude": 51.893921, "longitude": -8.475952},
    "C934": {"latitude": 51.888146, "longitude": -8.409735},
    "C935": {"latitude": 51.901362, "longitude": -8.473752},
    "C936": {"latitude": 51.900101, "longitude": -8.47064},
    "C937": {"latitude": 51.899352, "longitude": -8.473643},
    "C938": {"latitude": 51.901474, "longitude": -8.473892},
    "C939": {"latitude": 51.899929, "longitude": -8.473892},
    "C940": {"latitude": 51.919326782227, "longitude": -8.4175872802734},
    "C941": {"latitude": 51.91713, "longitude": -8.428848},
    "C942": {"latitude": 51.896942, "longitude": -8.49353},
    "C943": {"latitude": 51.895505, "longitude": -8.495449},
    "C944": {"latitude": 51.897158, "longitude": -8.468007},
    "C945": {"latitude": 51.889311, "longitude": -8.461533},
    "C946": {"latitude": 51.889458, "longitude": -8.459129},
    "C947": {"latitude": 51.898335, "longitude": -8.469969},
    "C948": {"latitude": 51.895505, "longitude": -8.495449},
    "C949": {"latitude": 51.894607543945, "longitude": -8.5013580322266},
    "C950": {"latitude": 51.906912, "longitude": -8.493669},
    "C951": {"latitude": 51.89323425293, "longitude": -8.4546661376953},
    "C952": {"latitude": 51.89613097469726, "longitude": -8.469903616237444},
    "C953": {"latitude": 51.891769, "longitude": -8.450546},
    "C954": {"latitude": 51.890024, "longitude": -8.394478},
    "C955": {"latitude": 51.887822, "longitude": -8.396907},
    "C956": {"latitude": 51.915894, "longitude": -8.440247},
    "C957": {"latitude": 51.917954, "longitude": -8.426514},
    "C958": {"latitude": 51.913833618164, "longitude": -8.4285736083984},
    "C959": {"latitude": 51.917953491211, "longitude": -8.4244537353516},
    "C960": {"latitude": 51.909715, "longitude": -8.517289},
    "C961": {"latitude": 51.899757, "longitude": -8.475609},
    "C962": {"latitude": 51.919128, "longitude": -8.481977},
    "C963": {"latitude": 51.878527, "longitude": -8.528097},
    "C964": {"latitude": 51.898727416992, "longitude": -8.3983612060547},
    "C965": {"latitude": 51.906416, "longitude": -8.352361},
    "C966": {"latitude": 51.907425, "longitude": -8.387375},
    "C967": {"latitude": 51.919326782227, "longitude": -8.3832550048828},
    "C968": {"latitude": 51.917953, "longitude": -8.399048},
    "C969": {"latitude": 51.912165811955326, "longitude": -8.45489231238302},
    "C970": {"latitude": 51.916282, "longitude": -8.47286},
}
def process_each(src_file_path, nodehex_mapping, rawcellid_to_nodehex, exclusion_list, missing_node_coords):
    df = pd.read_csv(src_file_path)
    
    for col in ['RSRQ', 'CQI', 'SNR', 'RSRP']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('-', np.nan).replace(r'^\s*$', np.nan, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()
    df.dropna(subset=['RSRQ', 'CQI', 'SNR', 'RSRP'], how='any', inplace=True)

    df['NODEHEX'] = df['NODEHEX'].astype(str).str.strip()
    if 'RAWCELLID' in df.columns:
        df['RAWCELLID'] = df['RAWCELLID'].astype(str).str.strip()
        df['NODEHEX'] = df['RAWCELLID'].map(rawcellid_to_nodehex).fillna(df['NODEHEX'])
        df = df[~df['RAWCELLID'].isin(exclusion_list)]
    
    df['node_Latitude'] = df['NODEHEX'].map(lambda x: nodehex_mapping[x]['latitude'] if x in nodehex_mapping else np.nan)
    df['node_Longitude'] = df['NODEHEX'].map(lambda x: nodehex_mapping[x]['longitude'] if x in nodehex_mapping else np.nan)
    
    valid_mask = df[['Latitude', 'Longitude', 'node_Latitude', 'node_Longitude']].notnull().all(axis=1)
    if valid_mask.any():
        user_lat = np.radians(df.loc[valid_mask, 'Latitude'].values)
        user_lon = np.radians(df.loc[valid_mask, 'Longitude'].values)
        node_lat = np.radians(df.loc[valid_mask, 'node_Latitude'].values)
        node_lon = np.radians(df.loc[valid_mask, 'node_Longitude'].values)
        dlat = node_lat - user_lat
        dlon = node_lon - user_lon
        a = np.sin(dlat / 2)**2 + np.cos(user_lat) * np.cos(node_lat) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df.loc[valid_mask, 'DFN'] = 6371.0 * c
    else:
        df['DFN'] = np.nan

    if 'RSRP' in df.columns and 'DFN' in df.columns:
        mask = ((df['RSRP'] >= -200) & (df['RSRP'] <= -180)) | (df['DFN'] > 4)
        df = df[~mask]
    
    if 'RAWCELLID' in df.columns:
        missing_mask = df[['node_Latitude', 'node_Longitude']].isnull().any(axis=1)
        if 'LACHEX' in df.columns:
            for _, row in df.loc[missing_mask, ['RAWCELLID', 'LACHEX']].dropna().iterrows():
                missing_node_coords.add((row['RAWCELLID'], row['LACHEX']))
        else:
            for rc in df.loc[missing_mask, 'RAWCELLID'].dropna().unique():
                missing_node_coords.add((rc, 'N/A'))
    
    DROP_COLUMNS = [col for col in 
        ['Operatorname', 'CellID', 'PINGAVG', 'PINGMIN', 'PINGMAX', 
        'PINGSTDEV', 'PINGLOSS', 'CELLHEX', 'RSSI', 'NRxRSRP',
        'NRxRSRQ', 'DL_bitrate', 'UL_bitrate', 'NetworkMode',
        'State', 'NODEHEX', 'LACHEX'] 
        if col in df.columns]
    if DROP_COLUMNS:
        df.drop(columns=DROP_COLUMNS, inplace=True)

    rename = {}
    if 'Longitude' in df.columns: rename['Longitude'] = 'user_Longitude'
    if 'Latitude' in df.columns: rename['Latitude'] = 'user_Latitude'
    if rename: df.rename(columns=rename, inplace=True)
    
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d_%H.%M.%S')
        except Exception:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        if df['Timestamp'].isnull().all():
            print(f"Warning: Could not parse Timestamp in {src_file_path}.")
        else:
            df['Day'] = df['Timestamp'].dt.weekday + 1
            df['time'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second
            df['isDriving'] = (df['Speed'] > 0).astype(int) if 'Speed' in df.columns else 0
    else:
        print(f"No Timestamp Column In {src_file_path}.")
    return df

def process_all_original(src_dir, nodehex_mapping, rawcellid_to_nodehex, exclusion_list, missing_node_coords):
    dfs = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = process_each(path, nodehex_mapping, rawcellid_to_nodehex, exclusion_list, missing_node_coords)
                    if df is not None and not df.empty:
                        dfs.append(df)
                        print(f"Included {path}")
                    else:
                        print(f"File {path} Made Empty DF.")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    return dfs

def combine_original_files(dfs):
    combined = pd.concat(dfs, ignore_index=True)
    if 'Timestamp' in combined.columns:
        combined['Timestamp'] = pd.to_datetime(combined['Timestamp'], errors='coerce')
        combined.sort_values(by='Timestamp', inplace=True)
    combined.to_csv(OUTPUT_DATASET, index=False)

if __name__ == "__main__":
    combined_dir = 'DATA/FULL_FEATURE_SET'
    CELLID_MAPPING = {
        '838355':'C500','842977':'C600','19756840':'C700','58139650':'C800',
        '843594':'C900','838538':'C901','844095':'C902','19736842':'C903',
        '124546049':'C904','10805003':'C905','11112973':'C906','11113741':'C907',
        '11115778':'C908','124546050':'C909','19747039':'C910','22695':'C911',
        '34056':'C912','34124':'C913','37174':'C914','57294850':'C915',
        '787673':'C916','788113':'C917','790203':'C918','790764':'C919',
        '790774':'C920','790885':'C921','795075':'C922','795085':'C923',
        '795114':'C924','795825':'C925','838303':'C926','838306':'C926',
        '838383':'C927','838385':'C928','838386':'C929','838388':'C930',
        '838394':'C931','838396':'C932','838398':'C933','838473':'C934',
        '838513':'C935','838514':'C936','838515':'C937','838516':'C938',
        '838518':'C939','838534':'C940','838535':'C941','838585':'C942',
        '838588':'C943','838614':'C944','838617':'C944','838644':'C945',
        '838647':'C946','838687':'C947','840683':'C948','840685':'C949',
        '843166':'C950','843263':'C951','843283':'C952','843315':'C953',
        '843334':'C954','843337':'C955','843423':'C956','843463':'C957',
        '843465':'C958','843466':'C959','843597':'C960','843617':'C961',
        '843644':'C962','843678':'C963','843827':'C964','843843':'C965',
        '843846':'C966','844093':'C967','844098':'C968','70430210':'C969',
        '70430211':'C970',
    }
    NODES_NOT_FOUND = [
        '0','19736841','19767337','19767039','19747337','19767033','10809091',
        '10858508','10863107','10863117','10944769','10944780','11016973',
        '11115266','11119115','11127051','19736590','19736609','19736777',
        '19736821','19736840','19737001','19737008','19737009','19737028',
        '19737029','19737033','19737071','19737337','19737339','19746590',
        '19746591','19746609','19746778','19746794','19746795','19746840',
        '19746841','19747033','19747163','19756590','19756609','19756794',
        '19756841','19756842','19757015','19757028','19757033','19757039',
        '19757163','19757217','19757337','19757339','19766590','19766711',
        '19766778','19766794','19766795','19766840','19766842','19767008',
        '19767015','19767027','19767070','19767339','788767','790153','790785',
        '838358','838519','838521','843318',
    ]
    missing_node_coords = set()
        
    dfs = process_all_original(ORIGINAL_DATASET_DIRECTORY, NODEHEX_MAPPING, CELLID_MAPPING, NODES_NOT_FOUND, missing_node_coords)
    combine_original_files(dfs)
    if missing_node_coords:
        print("Missing Node Coordinates:")
        for cid, lac in sorted(missing_node_coords):
            print(f"{cid} - {lac}")
    else:
        print("No Missing Node Coordinates.")
