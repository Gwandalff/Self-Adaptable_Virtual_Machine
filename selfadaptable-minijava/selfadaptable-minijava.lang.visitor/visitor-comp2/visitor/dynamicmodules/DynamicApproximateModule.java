package visitor.dynamicmodules;

import java.util.ArrayList;
import java.util.List;

import org.eclipse.xtext.EcoreUtil2;

import miniJava.visitor.miniJava.ArrayAccess;
import miniJava.visitor.miniJava.Assignee;
import miniJava.visitor.miniJava.Assignment;
import miniJava.visitor.miniJava.ForStatement;
import miniJava.visitor.miniJava.IntConstant;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Plus;
import miniJava.visitor.miniJava.Statement;
import miniJava.visitor.miniJava.SymbolRef;
import miniJava.visitor.miniJava.VariableDeclaration;
import visitor.AcceptInterface;
import visitor.IDynamicModule;

public class DynamicApproximateModule implements IDynamicModule {
	
	private List<Statement> arrayAssign;

	@Override
	public boolean updateBefore(AcceptInterface self) {
		if( complexVerification(self) ) {
			performOptimization(self);
		}
		self.detach(this);
		return false;
	}
	
	@Override
	public void updateAfter(AcceptInterface self) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int[] targetedNodes() {
		int[] out = {MiniJavaPackage.FOR_STATEMENT};
		return out;
	}
	
	@Override
	public Strategy callStrategy() {
		return Strategy.BEFORE;
	}
	
	private boolean complexVerification(AcceptInterface self) {
		try {
			ForStatement loop = ((ForStatement) self);
			String symbol = ((VariableDeclaration) loop.getDeclaration().getAssignee()).getName();
			
			boolean varUpdate = ((SymbolRef) loop.getProgression().getAssignee()).getSymbol().getName().equals(symbol);
			boolean constInc = (((Plus) loop.getProgression().getValue()).getRight() instanceof IntConstant &&
					((SymbolRef) ((Plus) loop.getProgression().getValue()).getLeft()).getSymbol().getName().equals(symbol)) ||
					(((Plus) loop.getProgression().getValue()).getLeft() instanceof IntConstant &&
					((SymbolRef) ((Plus) loop.getProgression().getValue()).getRight()).getSymbol().getName().equals(symbol));
			if(!(varUpdate && constInc)) return false;
			
			arrayAssign = new ArrayList();
			List<Statement> statements = loop.getBlock().getStatements();
			for (Statement st : statements) {
				if(st instanceof Assignment) {
					Assignee a = ((Assignment) st).getAssignee();
					if(a instanceof ArrayAccess && 
							((ArrayAccess) a).getIndex() instanceof SymbolRef &&
							((SymbolRef) ((ArrayAccess) a).getIndex()).getSymbol().getName().equals(symbol)) {
						arrayAssign.add(st);
					}
				}
			}
			
			return !arrayAssign.isEmpty();
		} catch (Exception e) {
			return false;
		}
	}
	
	private void performOptimization(AcceptInterface self) {
		ForStatement loop = ((ForStatement) self);
		Plus plus = ((Plus) loop.getProgression().getValue());
		int constIncr = 0;
		SymbolRef symbol = null;
		if(plus.getLeft() instanceof IntConstant) {
			constIncr = ((IntConstant) plus.getLeft()).getValue();
			((IntConstant) plus.getLeft()).setValue(constIncr*2);
			symbol = ((SymbolRef) plus.getRight());
		} else {
			constIncr = ((IntConstant) plus.getRight()).getValue();
			((IntConstant) plus.getRight()).setValue(constIncr*2);
			symbol = ((SymbolRef) plus.getLeft());
		}
		List<Statement> toAdd = new ArrayList<>();
		for (Statement st : loop.getBlock().getStatements()) {
			if(! arrayAssign.contains(st)) {
				Statement copy = EcoreUtil2.copy(st);
				List<SymbolRef> symbols = EcoreUtil2.getAllContentsOfType(copy, SymbolRef.class);
				for (SymbolRef sym : symbols) {
					if(sym.getSymbol() == symbol.getSymbol()) {
						IntConstant integer = MiniJavaFactory.eINSTANCE.createIntConstant();
						plus = MiniJavaFactory.eINSTANCE.createPlus();
						integer.setValue(constIncr);
						plus.setRight(integer);
						plus.setLeft(EcoreUtil2.copy(sym));
						EcoreUtil2.replace(sym, plus);
					}
				}
				toAdd.add(copy);
			} else {
				// create _+const for i+const
				IntConstant integer = MiniJavaFactory.eINSTANCE.createIntConstant();
				plus = MiniJavaFactory.eINSTANCE.createPlus();
				integer.setValue(constIncr);
				plus.setRight(integer);
				
				// copy assignment to avoid moving sub-tree
				Assignment copy = (Assignment) EcoreUtil2.copy(st);
				// save A[i] to put it on right side
				ArrayAccess pred = EcoreUtil2.copy(((ArrayAccess) copy.getAssignee()));
				// move symbol i from index to plus then put plus as index (i -> i+const)
				plus.setLeft(((ArrayAccess) copy.getAssignee()).getIndex());
				((ArrayAccess) copy.getAssignee()).setIndex(plus);
				// ... = A[i]
				copy.setValue(pred);
				toAdd.add(copy);
			}
		}
		loop.getBlock().getStatements().addAll(toAdd);
	}

}
