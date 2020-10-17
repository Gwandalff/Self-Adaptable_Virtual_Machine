package miniJava.interpreter.dynamicModules;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.xtext.EcoreUtil2;

import miniJava.interpreter.AbstractAdaptationModule;
import miniJava.interpreter.AdaptationContext;
import miniJava.interpreter.IAdaptationNode;
import miniJava.interpreter.decision.model.Goal;
import miniJava.interpreter.decision.model.ModelingElement.TAG;
import miniJava.interpreter.decision.model.Resource;
import miniJava.interpreter.decision.model.Softgoal;
import miniJava.interpreter.decision.model.Task;
import miniJava.interpreter.decision.model.Variable;
import miniJava.interpreter.miniJava.ArrayAccess;
import miniJava.interpreter.miniJava.Assignee;
import miniJava.interpreter.miniJava.Assignment;
import miniJava.interpreter.miniJava.Block;
import miniJava.interpreter.miniJava.ForStatement;
import miniJava.interpreter.miniJava.IntConstant;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.Minus;
import miniJava.interpreter.miniJava.Plus;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Statement;
import miniJava.interpreter.miniJava.SymbolRef;
import miniJava.interpreter.miniJava.Value;
import miniJava.interpreter.miniJava.VariableDeclaration;

public class ApproximateModule extends AbstractAdaptationModule {
	
	private List<Block> bodies;
	private Map<Block, List<Statement>> arraysAssign;
	private Map<Block, Block> arraysApproximate;
	private Task perforationRate;
	private Task accuracyImpact;
	private int count;
	private Variable smallPR;
	private Variable PR;
	private Variable bigPR;
	

	public ApproximateModule() {
		moduleTradeOff = new Goal("Approximate Trade-Off");
	}
	
	public boolean updateBefore(IAdaptationNode self, Object[] args) {
		if(bodies.contains(self)) {
			State state = (State) args[0];
			return performOptimization(self, state);
		}
		return true;
	}
	
	public boolean updateAfter(IAdaptationNode self, Object[] args, Value returned) {
		return false;
	}
	
	public Value byPassResult() {
		return null;
	}
	
	public Strategy callStrategy() {
		return Strategy.AROUND;
	}
	
	private boolean complexVerification(IAdaptationNode self) {
		try {
			ForStatement loop = ((ForStatement) self);
			String symbolName = ((VariableDeclaration) loop.getDeclaration().getAssignee()).getName();
			
			boolean varUpdate = ((SymbolRef) loop.getProgression().getAssignee()).getSymbol().getName().equals(symbolName);
			boolean constIncL = (((Plus) loop.getProgression().getValue()).getLeft() instanceof IntConstant &&
					((SymbolRef) ((Plus) loop.getProgression().getValue()).getRight()).getSymbol().getName().equals(symbolName));
			boolean constIncR = (((Plus) loop.getProgression().getValue()).getRight() instanceof IntConstant &&
					((SymbolRef) ((Plus) loop.getProgression().getValue()).getLeft()).getSymbol().getName().equals(symbolName));
			boolean constInc = constIncL || constIncR;
			if(!(varUpdate && constInc)) return false;
			
			Plus plus = (Plus) loop.getProgression().getValue();
			
			ArrayList<Statement> arrayAssign = new ArrayList<Statement>();
			Block body = loop.getBlock();

			List<Statement> statements = body.getStatements();
			for (Statement st : statements) {
				if(st instanceof Assignment) {
					Assignee a = ((Assignment) st).getAssignee();
					if(a instanceof ArrayAccess && 
							((ArrayAccess) a).getIndex() instanceof SymbolRef &&
							((SymbolRef) ((ArrayAccess) a).getIndex()).getSymbol().getName().equals(symbolName)) {
						arrayAssign.add(st);
					}
				}
			}
			if (!arrayAssign.isEmpty()) {
				bodies.add(body);
				arraysAssign.put(body, arrayAssign);
				int constIncr = 0;
				SymbolRef symbol = null;
				if(plus.getLeft() instanceof IntConstant) {
					constIncr = ((IntConstant) plus.getLeft()).getValue();
					symbol = ((SymbolRef) plus.getRight());
				} else {
					constIncr = ((IntConstant) plus.getRight()).getValue();
					symbol = ((SymbolRef) plus.getLeft());
				}
				List<Statement> toAdd = new ArrayList<>();
				for (Statement st : arrayAssign) {
					// create _+const for i+const
					IntConstant integer = MiniJavaFactory.eINSTANCE.createIntConstant();
					Minus minus = MiniJavaFactory.eINSTANCE.createMinus();
					integer.setValue(constIncr);
					minus.setRight(integer);
					minus.setLeft(EcoreUtil2.copy(symbol));
					// copy assignment to avoid moving sub-tree
					Assignment copy = (Assignment) EcoreUtil2.copy(st);
					// save A[i] to put it on right side
					ArrayAccess pred = (ArrayAccess) EcoreUtil2.copy(( copy.getAssignee()));
					// move symbol i from index to plus then put plus as index (i -> i-const)
					pred.setIndex(minus);
					copy.setValue(pred);
					// ... = A[i-const]
					toAdd.add(copy);
				}
				Block approximated = MiniJavaFactory.eINSTANCE.createBlock();
				approximated.getStatements().addAll(toAdd);
				arraysApproximate.put(body, approximated);
				return true;
			}
			return false;
		} catch (Exception e) {
			return false;
		}
	}
	
	private boolean performOptimization(IAdaptationNode self, State state) {
		count++;
		int step = 0;
		if(bigPR.value() != 0) {
			step = 8;
		} else if(PR.value() != 0) {
			step = 4;
		} else if(smallPR.value() != 0) {
			step = 2;
		} else {
			step = 1;
		}
//		System.out.println("STEP : " + step);
		if(count >= step) {
			count = 0;
			return true;
		}
		Block approximated = arraysApproximate.get((Block) self);
		approximated.evaluateStatement(state);
		
		return false;
	}

	public boolean isTargetedNode(IAdaptationNode s) {
		return complexVerification(s) || s instanceof Block;
	}

	@Override
	public void init(AdaptationContext adaptationContext) {
		if(bodies == null) {
			bodies = new ArrayList();
			arraysAssign = new HashMap();
			arraysApproximate = new HashMap();
		}
		
		count = 0;
		
		smallPR = new Variable("Small loop PR", 0, 0);
		PR = new Variable("loop PR", 0, 0);
		bigPR = new Variable("Big loop PR", 0, 0);
		
		perforationRate = new Task("Perforation Rate");
		accuracyImpact = new Task("Accuracy Impact");
		
		perforationRate.addContribution(smallPR, 1.0 / 7.0);
		perforationRate.addContribution(PR, 2.0 / 7.0);
		perforationRate.addContribution(bigPR, 4.0 / 7.0);
		
		accuracyImpact.addContribution(smallPR, 1.0 / 21.0);
		accuracyImpact.addContribution(PR, 4.0 / 21.0);
		accuracyImpact.addContribution(bigPR, 16.0 / 21.0);
	}

	@Override
	public void connectSoftGoal(Softgoal softgoal) {
		if (softgoal.ID.equals("Performance")) {
			softgoal.addContribution(perforationRate, 0.25);
		}
		if (softgoal.ID.equals("Accuracy")) {
			softgoal.addContribution(accuracyImpact, -0.75);
		}
	}

	@Override
	public void connectResource(Resource resource) {
		if(resource.ID.equals("CPU_PERCENT")) {
			smallPR.addTag(resource, TAG.SUPERIOR, 0.25);
			PR.addTag(resource, TAG.SUPERIOR, 0.5);
			bigPR.addTag(resource, TAG.SUPERIOR, 0.75);
		}
	}
	
	public void reset(AdaptationContext adaptationContext) {
		bodies = new ArrayList();
		arraysAssign = new HashMap();
		arraysApproximate = new HashMap();
		
		count = 0;
	}

}
