import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from urllib.parse import urlparse

df = pd.read_csv("geocitylinks.csv",header=None)
df = df.drop(columns=[0,1,4])
ids = ['http://geocities.com/azhim01/Aids.htm', 'http://www.geocities.com/aids_holisticspng/index.htm', 'http://geocities.com/Tokyo/6425/aids.htm', 'http://www.geocities.com/medipedia/17015.htm', 'http://geocities.com/aids_holisticspng/references.htm', 'http://geocities.com/HotSprings/3741/orglinks.html', 'http://geocities.com/HIV_and_AIDS/', 'http://us.geocities.com/btmbatha/aids2.html', 'http://geocities.com/WestHollywood/Heights/5010/aids.html', 'http://us.geocities.com/qapa.geo/aids.html', 'http://geocities.com/WestHollywood/5552/pagetwo.html', 'http://geocities.com/tlf_ph/network/network_archive_youthprimer.html', 'http://geocities.com/catolicas/articulos/actualidad/reports.htm', 'http://geocities.com/aids_holisticspng/series3.htm', 'http://geocities.com/mikelscott/081.htm', 'http://geocities.com/scfhivaids/homeframe.htm', 'http://geocities.com/kim1122b/AIDS.html', 'http://geocities.com/laineberman/aids_indonesia04.htm', 'http://geocities.com/tonawaparn/report.html', 'http://geocities.com/WestHollywood/5552/linksfor.htm', 'http://geocities.com/osduy_ngo/data.html', 'http://geocities.com/anionegraton/SouthPark.html', 'http://geocities.com/cuperuspe/webquest.html', 'http://geocities.com/oklarain/rainbowoct.html', 'http://www.hanmat.org/store/confident/golf/club/aids_crisis_in_africa.htm?t=rUepln-S9eIKEwjdi7vk0t6dAhUfH4cKHUdgECkQAhgBIAsw0--gAzgxUNPvoANQpPmYD1CJ76gQUOfh-BBQmt_5EFD50r8VUNvKkdkB&slt=8&slr=12&lpt=2', 'http://asia.geocities.com/sabljak007/AIDS11.htm', 'http://geocities.com/HotSprings/Villa/7435/internet.htm', 'http://geocities.com/athens/acropolis/9254/july01.html', 'http://geocities.com/HotSprings/3741/scilinks_en.html', 'http://www.geocities.com/aids_holisticspng/national_letters.htm', 'http://geocities.com/tlf_ph/network/network_health_aidsnews1207.html', 'http://geocities.com/doriendetombe/detombehandbook2003.html', 'http://geocities.com/ylhawks/Unresolved_World_ProblemsProject-aids.htm', 'http://geocities.com/johnpotterat/Newsarticles2-15.htm', 'http://geocities.com/ScaryAIDSFacts/', 'http://geocities.com/ashlyr1115/AshleySimpso1.html', 'http://geocities.com/Wellesley/8984/wom-aids.html', 'http://geocities.com/daveyboy_can/', 'http://www.google.ca/Top/Society/Issues/Health/Conditions_and_Diseases/Sexually_Transmitted_Diseases/AIDS/', 'http://geocities.com/popeducation/research1.html', 'http://www.geocities.com/zimbabwe_gurl/aidsedu.html', 'http://uk.geocities.com/ifeogo/religion_aids.html', 'http://geocities.com/guntoroutamadi/artikel-malam-renungan-aids.html', 'http://geocities.com/Yosemite/2772/FinalCapstone.htm', 'http://geocities.com/larrynjanet/disslarrych7', 'http://geocities.com/weinerap/duaids.htm', 'http://geocities.com/a280872/UA-AIDS.htm', 'http://geocities.com/dishdisho/aids.html', 'http://poz.com/', 'http://www.geocities.com/prostar_pendang/info_aids.html', 'http://geocities.com/otagogaily/back_issues/issue_58.htm', 'http://geocities.com/lisagarmire/Chp5WorksCited.htm', 'http://disabilityuk.com/masterpages/hiv/index.html', 'http://geocities.com/flippyflop3/testofhearts13.html', 'http://geocities.com/t_dambra/Statistics.html', 'http://geocities.com/chateau_gov/AIDS.html', 'http://us.geocities.com/marissa2/aidsresources.html', 'http://www.geocities.com/livhorg/PartnersResources.html', 'http://geocities.com/newsthai/week1800.html', 'http://geocities.com/afreval/documents/health_section.htm', 'http://geocities.com/deoshlok/aids.htm', 'http://www.aidslondon.com/', 'http://geocities.com/tlf_ph/network/network_archive_hivabtest.html', 'http://geocities.com/lamorena4rmsd/howaidsisspread.html', 'http://geocities.com/valestarplace/aidslinks.html', 'http://geocities.com/shireelyn/fight/fight.html', 'http://geocities.com/tlf_ph/network/network_health_aidsnews1205.html', 'http://geocities.com/mikelscott/035.htm', 'http://geocities.com/WestHollywood/2118/eaids2.html', 'http://www.drugabuse.gov/drugpages/HIV.html', 'http://us.geocities.com/pwacoalition/13-jail.htm', 'http://www.geocities.com/vampiricstudies/medical.html', 'http://geocities.com/opendoors_stl/links.htm', 'http://geocities.com/WestHollywood/4479/rainbo_resources.html', 'http://geocities.com/CapeCanaveral/Lab/1378/HIV_n_AIDS.html', 'http://geocities.com/HotSprings/9999/aidsandstdsresearches2tr.html', 'http://geocities.com/adrianchanwp/', 'http://geocities.com/msm_nopoliticalagenda/Pride03/Links.htm', 'http://geocities.com/gayankara/turkaids.htm', 'http://geocities.com/msm_nopoliticalagenda/WorldAIDS02/1.2.Programme.htm', 'http://geocities.com/HotSprings/5243/poster.htm', 'http://geocities.com/ninquiry2004/wanda/learning_activity.html', 'http://geocities.com/aidsprojectmpow/index.html', 'http://geocities.com/touchdpy/LinksPage.htm', 'http://geocities.com/WestHollywood/Castro/1521/aids.html', 'http://www.google.com.br/Top/Society/Issues/Health/Conditions_and_Diseases/Sexually_Transmitted_Diseases/AIDS/Workplace/', 'http://geocities.com/cspslibrary/seminars2002.html', 'http://geocities.com/TimesSquare/Lair/1217/aids-mulher.html', 'http://geocities.com/aids_youth/youth.html', 'http://uk.geocities.com/l0ox82d854/aids-impotence.html', 'http://geocities.com/Heartland/Pond/9005/TributeToDavid.html', 'http://us.geocities.com/quackwatch/hiv-aids.html', 'http://geocities.com/SouthBeach/Lagoon/2222/aids.htm', 'http://www.sonnyandmike.com/sonnytellsbrendastonehasaids95.htm', 'http://geocities.com/4birthing/HIV_Fact_Sheet_Chart.htm', 'http://geocities.com/vacune_hiv/acurarme7.html', 'http://geocities.com/queencaz2002/AIDSMAIN.html', 'http://geocities.com/bourbonstreet/porch/5017/aids.html', 'http://us.geocities.com/sciresquran/Articles/Quran_and_Scientific_Research.htm', 'http://geocities.com/sebaya01/Aids2.htm', 'http://geocities.com/Tokyo/Subway/6541/thang08-nam99/VN-new_AIDS_treatment.html', 'http://www.geocities.com/youth4sa/cubagays.html', 'http://geocities.com/herbal_hiv/Fag1.htm', 'http://geocities.com/lamorena4rmsd/howaidsstarted.html', 'http://geocities.com/vdcounsel/QUESTIONSandANSWERS.html', 'http://geocities.com/lizahayes23/magic-johnson-aids-story.html', 'http://geocities.com/elvis8377/What_is_AIDS_HIV_Hemophilia.html', 'http://geocities.com/SouthBeach/Pier/8747/Aids2.html', 'http://geocities.com/HotSprings/3741/newslinks_en.html', 'http://geocities.com/doriendetombe/detombecompramh8thesis.html', 'http://geocities.com/sigma_tf/HIV-AIDS.html']


pre_src = df[2].to_list()
pre_target = df[3].to_list()

src = []
for url in pre_src:
    parsed = urlparse(url)
    final_url = parsed.netloc + parsed.path.rsplit('/',1)[0]
    src.append(final_url)

target = []
for url in pre_target:
    parsed = urlparse(url)
    if parsed.netloc == 'geocities.com':
        final_url = parsed.netloc + parsed.path.rsplit('/',1)[0]
    else:
        final_url = parsed.netloc
    target.append(final_url)

nodes = pd.DataFrame(columns=['id','label','type'])
for url in src:
    if url not in nodes['id'].tolist():
        nodes = nodes.append({'id':url,'label':url,'type':1},ignore_index=True)

for url in target:
    if url not in nodes['id'].tolist():
        nodes = nodes.append({'id':url,'label':url,'type':0},ignore_index=True)
nodes.to_csv("nodes.csv",index=False)


dict = {"Source" : src, "Target" : target}
edges = pd.DataFrame(dict)
edges['Weight'] = 1
edges['Type'] = "directed"
edges = edges[edges["Target"] != '']
edges.to_csv('edges.csv',index=False)

'''
G.number_of_nodes()
for index, row in df.iterrows():
    edges.append(row[2],row[3],1,"directed")
G.number_of_edges()

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
nx.draw_networkx_labels(G, pos)
plt.savefig('foo.pdf')

nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
plt.show()

def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)
    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.savefig(file_name,bbox_inches="tight")
    del fig

#Assuming that the graph g has nodes and edges entered
save_graph(G,"foo.pdf")

nx.write_gexf(G, "test.gexf")

'''
