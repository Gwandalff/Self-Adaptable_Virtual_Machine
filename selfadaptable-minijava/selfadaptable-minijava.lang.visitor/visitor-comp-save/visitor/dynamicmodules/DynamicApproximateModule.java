package visitor.dynamicmodules;

import miniJava.visitor.miniJava.ForStatement;
import miniJava.visitor.miniJava.IntConstant;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Plus;
import visitor.AcceptInterface;
import visitor.IDynamicModule;

public class DynamicApproximateModule implements IDynamicModule {

	@Override
	public boolean update(AcceptInterface self) {
		System.out.println("Call module");
		if( complexVerification(self) ) {
			performOptimization(self);
		}
		System.out.println("End module");
		return false;
	}

	@Override
	public int[] targetedNodes() {
		int[] out = new int[1];
		out[0] = MiniJavaPackage.FOR_STATEMENT;
		return out;
	}
	
	private boolean complexVerification(AcceptInterface self) {
		try {
			ForStatement loop = ((ForStatement) self);
			return ((Plus) loop.getProgression().getValue()).getRight() instanceof IntConstant;
		} catch (Exception e) {
			return false;
		}
	}
	
	private void performOptimization(AcceptInterface self) {
		System.out.println("Optimization");
	}

}
