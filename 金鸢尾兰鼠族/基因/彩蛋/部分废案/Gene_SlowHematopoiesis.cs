using RimWorld;
using Verse;

namespace OberoniaAureaGene;

//造血缓慢
public class Gene_SlowHematopoiesis : Gene
{
    private static readonly float BloodLossAdjuest = 0.00002666666f;
    public override void Tick()
    {
        if (pawn.IsHashIntervalTick(250))
        {
            AdjuestBloodLoss(pawn);
        }
    }
    private static void AdjuestBloodLoss(Pawn pawn)
    {
        Hediff bloodLoss = pawn.health.hediffSet.GetFirstHediffOfDef(HediffDefOf.BloodLoss);
        if (bloodLoss != null)
        {
            bloodLoss.Severity += BloodLossAdjuest;
        }
    }
}
