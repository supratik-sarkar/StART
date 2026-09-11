import type { EdaProfile, ScenarioItem } from '../../contracts/types'
import { Bars, CopyValue, DataBlock, Empty, Matrix, ScoreStrip, Section, ScientificTable } from '../investigation/Science'
export function EdaCanvas({profile,activeScenario,loading,error}: {profile:EdaProfile|null;activeScenario:ScenarioItem|null;loading:boolean;error:string|null;onOpenTestCatalog:()=>void}) {
  if(error)return <div className="eda-review"><p role="alert">{error}</p></div>
  if(loading)return <div className="eda-review"><Empty title="Inspecting dataset">Loading descriptive profiling.</Empty></div>
  if(!profile)return null
  const data=profile.data_profile ?? profile
  return <div className="eda-review"><div className="eyebrow">DATA INSPECTION / DEFINE</div><h1>{activeScenario?.label ?? 'Dataset profile'}</h1><p className="quiet">Descriptive inspection · before model evaluation</p><Section title="Dataset overview"><ScoreStrip data={data.overview ?? {}}/></Section><Section title="Class balance"><Bars data={data.target_distribution?.counts ?? {}} name="Class observations"/></Section><Section title="Feature schema"><ScientificTable rows={data.schema ?? []}/></Section><details><summary>Descriptive moments</summary><ScientificTable rows={data.feature_moments ?? []}/></details><Section title="Feature correlation"><Matrix data={data.correlation?.matrix} columns={data.correlation?.columns}/></Section><DataBlock title="Model context" data={profile.model_fixture_context}/><DataBlock title="Portfolio composition" data={profile.portfolio_composition}/><Section title="Data provenance"><p>{profile.provenance?.provenance_note}</p>{profile.provenance?.sha256_fingerprint&&<CopyValue value={profile.provenance.sha256_fingerprint} truncate/>}</Section></div>
}
