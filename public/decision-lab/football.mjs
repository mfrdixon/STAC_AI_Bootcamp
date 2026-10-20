export const footballQuestion={entities:['New York Yankees','New England Patriots'],metric:'campaign_interest',period:'fictional-campaign-period',population:'target campaign audience'};
export const footballSources=[
 {id:'S1',entity:'New York Yankees',metric:'campaign_interest',period:'fictional-campaign-period',population:'target campaign audience',selection:'documented_sample',title:'Yankees synthetic audience records'},
 {id:'S2',entity:'New England Patriots',metric:'campaign_interest',period:'fictional-campaign-period',population:'target campaign audience',selection:'documented_sample',title:'Patriots synthetic audience records'},
 {id:'S3',entity:'New York Yankees',metric:'fan_poll',period:'fictional-campaign-period',population:'voluntary fan-site respondents',selection:'voluntary',title:'Yankees fan-site popularity poll'},
 {id:'S4',entity:'New England Patriots',metric:'campaign_interest',period:'historical-period',population:'target campaign audience',selection:'documented_sample',title:'Patriots results from a different period'}
];
// Deliberately constructed Simpson reversal: not actual survey results; wins denotes positive responses for the generic binomial helper.
export const footballGroups=[
 {entity:'New York Yankees',stratum:'local audience',n:12,wins:9},
 {entity:'New York Yankees',stratum:'national audience',n:5,wins:1},
 {entity:'New England Patriots',stratum:'local audience',n:5,wins:4},
 {entity:'New England Patriots',stratum:'national audience',n:12,wins:3}
];
