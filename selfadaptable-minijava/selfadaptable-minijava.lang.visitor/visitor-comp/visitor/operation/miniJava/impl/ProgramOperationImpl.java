package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Program;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.TypeDeclaration;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.ProgramOperation;

public class ProgramOperationImpl implements ProgramOperation {
	private final Program it;

	private final VisitorInterface vis;

	public ProgramOperationImpl(Program it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public void main() {
		((ProgramOperation)this.it.accept(vis)).initialize( (null));
		((ProgramOperation)this.it.accept(vis)).execute();
	}

	public void initialize(EList args) {
		Context rootCont = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
		State state = ((State) (MiniJavaFactory.eINSTANCE.createState()));
		state.setOutputStream(MiniJavaFactory.eINSTANCE.createOutputStream());
		state.setRootFrame(MiniJavaFactory.eINSTANCE.createFrame());
		state.getRootFrame().setRootContext(rootCont);
		this.it.setState(state);
	}

	public State execute() {
		State result;
		Method main = ((Method) (((ProgramOperation)this.it.accept(vis)).findMain()));
		((BlockOperation)main.getBody().accept(vis)).evaluateStatementKeepContext((State) (this.it.getState()));
		result = ((State) (this.it.getState()));
		return result;
	}

	public Method findMain() {
		Method result;
		result = ((Method) (null));
		for(TypeDeclaration clazz: this.it.getClasses()) {
			for(Member member: clazz.getMembers()) {
				if(member instanceof Method) {
					Method method = ((Method) (member));
					if(EqualService.equals((method.getName()), ("main"))) {
						result = ((Method) (method));
					}
				}
			}
		}
		return result;
	}
}
